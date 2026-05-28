#!/usr/bin/env python3
"""
run_code.py

Measure original-span preference on paired code counterfactual datasets,
such as QuixBugs or DebugBench.

For each row:

  original_buggy_program       contains the original buggy span
  counterfactual_buggy_program contains a perturbed buggy span

The script computes negative log likelihood only on the changed span.

Main score:

  delta_sum_nll =
      NLL(counterfactual_span in counterfactual_buggy_program)
    - NLL(original_span in original_buggy_program)

Interpretation:

  delta_sum_nll > 0
      model assigns higher probability to the original buggy span

  delta_sum_nll < 0
      model assigns higher probability to the counterfactual buggy span

Expected input JSONL fields:

  original_buggy_program
  counterfactual_buggy_program
  buggy_span or original_span
  counterfactual_span

Optional fields used for more reliable span location:

  buggy_line
  counterfactual_line
  span_start
  span_end
  line_index

Examples:

CUDA_VISIBLE_DEVICES=0 python formal_released/scripts/run_code.py \
  --model bigcode/starcoderbase-3b \
  --input_jsonl generated/quixbugs_manual_counterfactual_full.jsonl \
  --out_jsonl result_code/quixbugs_starcoderbase3b_spanlogprob.jsonl

CUDA_VISIBLE_DEVICES=0 python formal_released/scripts/run_code.py \
  --model bigcode/starcoderbase-3b \
  --input_jsonl generated_debugbench/debugbench_counterfactual_curated_llm_fixed43.jsonl \
  --out_jsonl result_code/debugbench_starcoderbase3b_spanlogprob.jsonl
"""

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model", required=True)
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--out_summary", default=None)

    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--device_map", default="auto")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--cache_dir", default=None)

    p.add_argument(
        "--skip_token_count_mismatch",
        action="store_true",
        help="Skip examples where original and counterfactual spans tokenize to different lengths.",
    )

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


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def find_unique(haystack: str, needle: str) -> Optional[int]:
    first = haystack.find(needle)
    if first == -1:
        return None
    second = haystack.find(needle, first + 1)
    if second != -1:
        return None
    return first


def line_offsets(text: str) -> Tuple[List[str], List[int]]:
    lines = text.splitlines(keepends=True)
    offsets = []
    cur = 0
    for line in lines:
        offsets.append(cur)
        cur += len(line)
    return lines, offsets


def find_occurrences(s: str, sub: str) -> List[int]:
    out = []
    start = 0
    while True:
        i = s.find(sub, start)
        if i == -1:
            break
        out.append(i)
        start = i + 1
    return out


def locate_span(
    text: str,
    span: str,
    line: Optional[str] = None,
    span_start: Optional[int] = None,
    span_end: Optional[int] = None,
    line_index: Optional[int] = None,
) -> Tuple[Optional[int], Optional[int], str]:
    """
    More robust span locator.

    Priority:
      1. explicit absolute span_start, but allow span_end to change
      2. line_index + preferred column
      3. exact line + closest column
      4. unique full-text fallback
    """

    # 1. Explicit absolute start. Important when replacement length differs.
    if span_start is not None:
        if 0 <= span_start <= len(text):
            end = span_start + len(span)
            if end <= len(text) and text[span_start:end] == span:
                return span_start, end, "explicit_start_len"

    # Prepare line offsets.
    lines, offsets = line_offsets(text)

    preferred_col = None
    if span_start is not None and line_index is not None:
        if 0 <= line_index < len(offsets):
            preferred_col = span_start - offsets[line_index]

    # 2. Use line_index if available.
    if line_index is not None and 0 <= line_index < len(lines):
        target_line = lines[line_index]
        occs = find_occurrences(target_line, span)

        if occs:
            if preferred_col is not None:
                # Pick occurrence closest to original column.
                best = min(occs, key=lambda x: abs(x - preferred_col))
            elif len(occs) == 1:
                best = occs[0]
            else:
                best = occs[0]

            start = offsets[line_index] + best
            end = start + len(span)
            return start, end, "line_index_closest_span"

    # 3. Exact line fallback.
    if line:
        line_positions = find_occurrences(text, line)
        if line_positions:
            # Prefer exact unique line; otherwise first line occurrence.
            line_start = line_positions[0]
            occs = find_occurrences(line, span)

            if occs:
                if preferred_col is not None:
                    best = min(occs, key=lambda x: abs(x - preferred_col))
                elif len(occs) == 1:
                    best = occs[0]
                else:
                    best = occs[0]

                start = line_start + best
                end = start + len(span)
                return start, end, "line_text_closest_span"

    # 4. Unique full-text fallback.
    full_occs = find_occurrences(text, span)
    if len(full_occs) == 1:
        start = full_occs[0]
        return start, start + len(span), "full_text_unique_span"

    return None, None, "not_found_or_ambiguous"


def token_indices_for_char_span(
    offsets: List[Tuple[int, int]],
    char_start: int,
    char_end: int,
) -> List[int]:
    idxs = []
    for i, (a, b) in enumerate(offsets):
        if a == b:
            continue
        if b <= char_start:
            continue
        if a >= char_end:
            break
        if max(a, char_start) < min(b, char_end):
            idxs.append(i)
    return idxs


@torch.no_grad()
def token_nlls_with_offsets(model, tokenizer, text: str, device, max_length: int):
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
        return offsets, [float("nan")] * input_ids.shape[1], input_ids[0].detach().cpu().tolist()

    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        reduction="none",
    )

    # nlls[i] = NLL of token i predicted from tokens before it.
    # nlls[0] is undefined.
    nlls = [float("nan")] + token_loss.detach().cpu().tolist()
    ids = input_ids[0].detach().cpu().tolist()

    return offsets, nlls, ids


def score_span(
    model,
    tokenizer,
    text: str,
    span_start: int,
    span_end: int,
    device,
    max_length: int,
) -> Dict[str, Any]:
    offsets, nlls, input_ids = token_nlls_with_offsets(
        model=model,
        tokenizer=tokenizer,
        text=text,
        device=device,
        max_length=max_length,
    )

    idxs = token_indices_for_char_span(offsets, span_start, span_end)

    vals = []
    valid_idxs = []
    for i in idxs:
        if i < len(nlls) and math.isfinite(nlls[i]):
            vals.append(float(nlls[i]))
            valid_idxs.append(i)

    if not vals:
        return {
            "sum_nll": None,
            "mean_nll": None,
            "num_tokens": 0,
            "token_indices": idxs,
            "tokens": [],
        }

    tokens = tokenizer.convert_ids_to_tokens([input_ids[i] for i in valid_idxs])

    return {
        "sum_nll": float(sum(vals)),
        "mean_nll": float(sum(vals) / len(vals)),
        "num_tokens": int(len(vals)),
        "token_indices": valid_idxs,
        "tokens": tokens,
    }


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    def collect(key):
        xs = []
        for r in records:
            v = r.get(key)
            if v is not None and math.isfinite(float(v)):
                xs.append(float(v))
        return xs

    def stats(xs):
        if not xs:
            return {}
        arr = np.array(xs, dtype=np.float64)
        return {
            "n": int(len(arr)),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "frac_positive": float((arr > 0).mean()),
            "positive_mean": float(arr[arr > 0].mean()) if np.any(arr > 0) else None,
            "positive_median": float(np.median(arr[arr > 0])) if np.any(arr > 0) else None,
        }

    return {
        "delta_sum_nll": stats(collect("delta_sum_nll")),
        "delta_mean_nll": stats(collect("delta_mean_nll")),
        "num_records": len(records),
        "num_token_count_mismatch": sum(
            1 for r in records
            if r.get("orig_num_tokens") != r.get("cf_num_tokens")
        ),
        "num_positive": sum(
            1 for r in records
            if r.get("delta_sum_nll") is not None and r["delta_sum_nll"] > 0
        ),
    }


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    if args.out_summary is None:
        args.out_summary = args.out_jsonl.replace(".jsonl", ".summary.json")

    print("[load rows]", args.input_jsonl)
    rows = read_jsonl(args.input_jsonl)
    print("[num rows]", len(rows))

    print("[load model]", args.model)

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

    out_records = []
    skipped = []

    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for row in tqdm(rows, desc="scoring spans"):
            name = row.get("name", "")

            orig_text = row.get("original_buggy_program")
            cf_text = row.get("counterfactual_buggy_program")

            if orig_text is None or cf_text is None:
                skipped.append({
                    "name": name,
                    "reason": "missing original_buggy_program or counterfactual_buggy_program",
                })
                continue

            orig_span = row.get("buggy_span", row.get("original_span"))
            cf_span = row.get("counterfactual_span")

            if not orig_span or not cf_span:
                skipped.append({
                    "name": name,
                    "reason": "missing buggy_span/original_span or counterfactual_span",
                })
                continue

            orig_line = row.get("buggy_line")
            cf_line = row.get("counterfactual_line")

            span_start = row.get("span_start", None)
            span_end = row.get("span_end", None)
            line_index = row.get("line_index", None)

            orig_s, orig_e, orig_loc_mode = locate_span(
                text=orig_text,
                span=orig_span,
                line=orig_line,
                span_start=span_start,
                span_end=span_end,
                line_index=line_index,
            )

            cf_s, cf_e, cf_loc_mode = locate_span(
                text=cf_text,
                span=cf_span,
                line=cf_line,
                span_start=span_start,
                span_end=span_start + len(cf_span) if span_start is not None else None,
                line_index=line_index,
            )

            if orig_s is None or cf_s is None:
                skipped.append({
                    "name": name,
                    "reason": "could_not_locate_span",
                    "orig_loc_mode": orig_loc_mode,
                    "cf_loc_mode": cf_loc_mode,
                    "orig_span": orig_span,
                    "cf_span": cf_span,
                })
                continue

            orig_score = score_span(
                model=model,
                tokenizer=tokenizer,
                text=orig_text,
                span_start=orig_s,
                span_end=orig_e,
                device=device,
                max_length=args.max_length,
            )

            cf_score = score_span(
                model=model,
                tokenizer=tokenizer,
                text=cf_text,
                span_start=cf_s,
                span_end=cf_e,
                device=device,
                max_length=args.max_length,
            )

            if orig_score["sum_nll"] is None or cf_score["sum_nll"] is None:
                skipped.append({
                    "name": name,
                    "reason": "empty_or_unscorable_span",
                    "orig_span": orig_span,
                    "cf_span": cf_span,
                    "orig_score": orig_score,
                    "cf_score": cf_score,
                })
                continue

            token_count_mismatch = orig_score["num_tokens"] != cf_score["num_tokens"]

            if args.skip_token_count_mismatch and token_count_mismatch:
                skipped.append({
                    "name": name,
                    "reason": "token_count_mismatch",
                    "orig_num_tokens": orig_score["num_tokens"],
                    "cf_num_tokens": cf_score["num_tokens"],
                    "orig_span": orig_span,
                    "cf_span": cf_span,
                })
                continue

            delta_sum = cf_score["sum_nll"] - orig_score["sum_nll"]
            delta_mean = cf_score["mean_nll"] - orig_score["mean_nll"]

            rec = {
                "name": name,
                "variant_id": row.get("variant_id", None),

                "buggy_span": orig_span,
                "fix_span": row.get("fix_span", None),
                "counterfactual_span": cf_span,

                "buggy_line": orig_line,
                "counterfactual_line": cf_line,
                "solution_line": row.get("solution_line", None),

                "orig_span_start": orig_s,
                "orig_span_end": orig_e,
                "cf_span_start": cf_s,
                "cf_span_end": cf_e,

                "orig_loc_mode": orig_loc_mode,
                "cf_loc_mode": cf_loc_mode,

                "orig_sum_nll": orig_score["sum_nll"],
                "cf_sum_nll": cf_score["sum_nll"],
                "delta_sum_nll": delta_sum,

                "orig_mean_nll": orig_score["mean_nll"],
                "cf_mean_nll": cf_score["mean_nll"],
                "delta_mean_nll": delta_mean,

                "orig_num_tokens": orig_score["num_tokens"],
                "cf_num_tokens": cf_score["num_tokens"],
                "token_count_mismatch": token_count_mismatch,

                "orig_tokens": orig_score["tokens"],
                "cf_tokens": cf_score["tokens"],

                "mutation_type": row.get("mutation_type", None),

                # Keep full programs for inspection.
                "original_buggy_program": orig_text,
                "counterfactual_buggy_program": cf_text,
                "solution": row.get("solution", None),
            }

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_records.append(rec)

    summary = summarize(out_records)
    summary.update({
        "model": args.model,
        "input_jsonl": args.input_jsonl,
        "out_jsonl": args.out_jsonl,
        "num_input_rows": len(rows),
        "num_scored_rows": len(out_records),
        "num_skipped": len(skipped),
        "skip_token_count_mismatch": args.skip_token_count_mismatch,
    })

    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    skipped_path = args.out_jsonl.replace(".jsonl", ".skipped.jsonl")
    with open(skipped_path, "w", encoding="utf-8") as f:
        for r in skipped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("[DONE]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[skipped]", skipped_path)


if __name__ == "__main__":
    main()