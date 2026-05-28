#!/usr/bin/env python3
"""
run_math.py

Measure original-number preference on paired math perturbation datasets.

For each paired problem:
    original_problem -> problem

We extract numeric spans from both versions, align them by order, keep changed
number pairs, and compute NLL only on those number tokens.

Main score:
    delta_sum_nll = NLL(perturbed number tokens) - NLL(original number tokens)

Interpretation:
    delta_sum_nll > 0
        => perturbed number is less likely
        => model prefers original number

Example:
CUDA_VISIBLE_DEVICES=0 python formal_released/scripts/run_math.py \
  --model EleutherAI/pythia-1.4b \
  --levels "Level 1,Level 2,Level 3" \
  --out_jsonl result_num/mathperturb200_pythia14b_L1L3_numlogprob.jsonl

CUDA_VISIBLE_DEVICES=0 python formal_released/scripts/run_math.py \
  --model Qwen/Qwen2.5-1.5B \
  --levels "Level 1,Level 2,Level 3" \
  --out_jsonl result_num/mathperturb200_qwen15b_L1L3_numlogprob.jsonl

CUDA_VISIBLE_DEVICES=0 python formal_released/scripts/run_math.py \
  --model YU-MO/Yumo-nano \
  --dataset generated/manual_natural_number_perturbed_52.jsonl \
  --levels all \
  --out_jsonl result_num/manual_natural_yumo_numlogprob.jsonl
"""

import argparse
import json
import math
import os
import re
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# Captures common math numeric forms:
# integers, decimals, signed numbers, percentages, simple LaTeX fractions.
NUMBER_RE = re.compile(
    r"""
    (?:
        \\frac\s*\{\s*[-+]?\d+(?:\.\d+)?\s*\}\s*\{\s*[-+]?\d+(?:\.\d+)?\s*\}
        |
        (?<![A-Za-z])
        [-+]?\d+(?:\.\d+)?%?
        (?![A-Za-z])
    )
    """,
    re.VERBOSE,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="stellaathena/math_perturbed_200")
    p.add_argument("--split", default="test")
    p.add_argument("--max_samples", type=int, default=None)

    # Added: level filtering
    p.add_argument(
        "--levels",
        type=str,
        default="Level 1,Level 2,Level 3",
        help=(
            "Comma-separated levels to keep, e.g. 'Level 1,Level 2,Level 3'. "
            "Use --levels all to disable level filtering."
        ),
    )

    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--device_map", default="auto")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--cache_dir", default=None)

    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--out_summary", default=None)
    return p.parse_args()


def get_dtype(name: str):
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(name)


def extract_numbers(text: str) -> List[Tuple[int, int, str]]:
    out = []
    for m in NUMBER_RE.finditer(text):
        s, e = m.span()
        val = m.group()
        out.append((s, e, val))
    return out


def normalize_num_string(s: str) -> str:
    return re.sub(r"\s+", "", s.strip())


def align_number_spans_by_order(
    orig_nums: List[Tuple[int, int, str]],
    pert_nums: List[Tuple[int, int, str]],
) -> List[Dict[str, Any]]:
    """
    Simple alignment by order.

    For this dataset, the perturbations are usually template-preserving, so
    order alignment is a reasonable first pass.

    If counts differ, align up to min length and flag the mismatch.
    """
    n = min(len(orig_nums), len(pert_nums))
    aligned = []

    count_mismatch = len(orig_nums) != len(pert_nums)

    for i in range(n):
        os_, oe, ov = orig_nums[i]
        ps, pe, pv = pert_nums[i]

        changed = normalize_num_string(ov) != normalize_num_string(pv)

        aligned.append({
            "num_index": i,
            "orig_start": os_,
            "orig_end": oe,
            "orig_num": ov,
            "pert_start": ps,
            "pert_end": pe,
            "pert_num": pv,
            "changed": changed,
            "count_mismatch": count_mismatch,
            "orig_num_count": len(orig_nums),
            "pert_num_count": len(pert_nums),
        })

    return aligned


@torch.no_grad()
def token_nlls_with_offsets(model, tokenizer, text: str, device, max_length: int):
    """
    Returns:
        offsets: list[(char_start, char_end)] for each token
        nlls:    list[float], token-level next-token NLL

    nlls[t] is the NLL of token t predicted from previous tokens.
    nlls[0] is NaN.
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    input_ids = enc["input_ids"].to(device)
    offsets = enc["offset_mapping"][0].tolist()

    if input_ids.shape[1] < 2:
        return offsets, [float("nan")] * input_ids.shape[1]

    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        reduction="none",
    )

    nlls = [float("nan")] + token_loss.detach().cpu().tolist()
    return offsets, nlls


def span_token_indices(offsets: List[Tuple[int, int]], start: int, end: int) -> List[int]:
    idxs = []
    for i, (a, b) in enumerate(offsets):
        # skip special empty-offset tokens if any
        if a == b:
            continue
        if b <= start:
            continue
        if a >= end:
            break
        if max(a, start) < min(b, end):
            idxs.append(i)
    return idxs


def score_span_from_cached(
    offsets: List[Tuple[int, int]],
    nlls: List[float],
    start: int,
    end: int,
):
    idxs = span_token_indices(offsets, start, end)
    vals = [nlls[i] for i in idxs if i < len(nlls) and math.isfinite(nlls[i])]

    if not vals:
        return {
            "mean_nll": None,
            "sum_nll": None,
            "num_tokens": 0,
            "token_indices": idxs,
        }

    return {
        "mean_nll": float(sum(vals) / len(vals)),
        "sum_nll": float(sum(vals)),
        "num_tokens": len(vals),
        "token_indices": idxs,
    }


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    def collect(key):
        return [r[key] for r in records if r.get(key) is not None and math.isfinite(r[key])]

    def stats(xs):
        if not xs:
            return {}
        t = torch.tensor(xs, dtype=torch.float32)
        return {
            "n": len(xs),
            "mean": float(t.mean()),
            "median": float(t.median()),
            "std": float(t.std(unbiased=False)),
            "min": float(t.min()),
            "max": float(t.max()),
            "frac_positive": float((t > 0).float().mean()),
        }

    by_type = {}
    for r in records:
        typ = str(r.get("type", "unknown"))
        by_type.setdefault(typ, []).append(r["delta_sum_nll"])

    by_level = {}
    for r in records:
        lvl = str(r.get("level", "unknown"))
        by_level.setdefault(lvl, []).append(r["delta_sum_nll"])

    return {
        "delta_sum_nll": stats(collect("delta_sum_nll")),
        "delta_mean_nll": stats(collect("delta_mean_nll")),
        "by_type_delta_sum_nll": {k: stats(v) for k, v in sorted(by_type.items())},
        "by_level_delta_sum_nll": {k: stats(v) for k, v in sorted(by_level.items())},
        "num_records": len(records),
        "num_count_mismatch_records": sum(1 for r in records if r.get("count_mismatch")),
    }


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    if args.out_summary is None:
        args.out_summary = args.out_jsonl.replace(".jsonl", ".summary.json")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=get_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
    )
    model.eval()
    device = next(model.parameters()).device

    print(f"[load dataset] {args.dataset} :: {args.split}")
    if args.dataset.endswith(".jsonl") or args.dataset.endswith(".json"):
        ds = load_dataset(
            "json",
            data_files=args.dataset,
            split="train",
            cache_dir=args.cache_dir,
        )
    else:
        ds = load_dataset(
            args.dataset,
            split=args.split,
            cache_dir=args.cache_dir,
        )
    print("[columns]", ds.column_names)
    print("[rows before level filter]", len(ds))

    # Added: keep only selected levels, default Level 1-3
    if args.levels is not None and args.levels.strip().lower() != "all":
        allowed_levels = {x.strip() for x in args.levels.split(",") if x.strip()}

        if "level" not in ds.column_names:
            raise ValueError(
                f"--levels was provided, but dataset has no 'level' column. "
                f"Available columns: {ds.column_names}"
            )

        before = len(ds)
        ds = ds.filter(lambda r: str(r["level"]).strip() in allowed_levels)
        print(f"[after level filter {sorted(allowed_levels)}] {before} -> {len(ds)}")
    else:
        print("[level filter disabled] using all levels")

    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
        print("[after max_samples]", len(ds))

    records = []

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for row in tqdm(ds):
            idx = row.get("idx", None)
            original_problem = str(row["original_problem"])
            perturbed_problem = str(row["problem"])

            orig_nums = extract_numbers(original_problem)
            pert_nums = extract_numbers(perturbed_problem)

            aligned = align_number_spans_by_order(orig_nums, pert_nums)
            changed_pairs = [a for a in aligned if a["changed"]]

            if not changed_pairs:
                continue

            # Add a stable prefix so the first token of a number is not at position 0.
            prefix = "Problem: "
            orig_text = prefix + original_problem
            pert_text = prefix + perturbed_problem

            orig_offsets, orig_nlls = token_nlls_with_offsets(
                model, tokenizer, orig_text, device=device, max_length=args.max_length
            )
            pert_offsets, pert_nlls = token_nlls_with_offsets(
                model, tokenizer, pert_text, device=device, max_length=args.max_length
            )

            for pair in changed_pairs:
                orig_s = len(prefix) + pair["orig_start"]
                orig_e = len(prefix) + pair["orig_end"]
                pert_s = len(prefix) + pair["pert_start"]
                pert_e = len(prefix) + pair["pert_end"]

                orig_score = score_span_from_cached(orig_offsets, orig_nlls, orig_s, orig_e)
                pert_score = score_span_from_cached(pert_offsets, pert_nlls, pert_s, pert_e)

                if orig_score["sum_nll"] is None or pert_score["sum_nll"] is None:
                    continue

                delta_sum = pert_score["sum_nll"] - orig_score["sum_nll"]
                delta_mean = pert_score["mean_nll"] - orig_score["mean_nll"]

                rec = {
                    "idx": idx,
                    "type": row.get("type", None),
                    "level": row.get("level", None),

                    "orig_num": pair["orig_num"],
                    "pert_num": pair["pert_num"],
                    "num_index": pair["num_index"],
                    "count_mismatch": pair["count_mismatch"],
                    "orig_num_count": pair["orig_num_count"],
                    "pert_num_count": pair["pert_num_count"],

                    "orig_sum_nll": orig_score["sum_nll"],
                    "pert_sum_nll": pert_score["sum_nll"],
                    "delta_sum_nll": delta_sum,

                    "orig_mean_nll": orig_score["mean_nll"],
                    "pert_mean_nll": pert_score["mean_nll"],
                    "delta_mean_nll": delta_mean,

                    "orig_num_tokens": orig_score["num_tokens"],
                    "pert_num_tokens": pert_score["num_tokens"],

                    "original_problem": original_problem,
                    "perturbed_problem": perturbed_problem,
                    "original_answer": row.get("original_answer", None),
                    "answer": row.get("answer", None),
                }

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                records.append(rec)

    summary = summarize(records)
    summary.update({
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "levels": args.levels,
        "out_jsonl": args.out_jsonl,
    })

    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[DONE]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()