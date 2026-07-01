#!/usr/bin/env python3
"""
Compute NLL only for the changed word/token span.

Expected JSONL fields:
  prefix
  target
  substitute   # or selected_candidate
  suffix

For each row:
  original  = prefix + target + suffix
  perturbed = prefix + substitute + suffix

Output includes:
  orig_changed_sum_nll
  pert_changed_sum_nll
  delta_sum_nll = pert_changed_sum_nll - orig_changed_sum_nll

Positive delta means the model assigns lower NLL / higher likelihood to the original word.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def overlap(a0, a1, b0, b1):
    return max(a0, b0) < min(a1, b1)


def get_changed_token_indices(tokenizer, text, span_start, span_end):
    """
    Return token indices whose character offsets overlap the changed word span.

    This is safer than using len(tokenizer(prefix)) because BPE tokenization can
    merge whitespace and words across the prefix boundary.
    """
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = enc["offset_mapping"]

    indices = []
    for i, (s, e) in enumerate(offsets):
        if s == e:
            continue
        if overlap(s, e, span_start, span_end):
            indices.append(i)

    return indices, enc["input_ids"], offsets


@torch.no_grad()
def changed_span_nll(
    model,
    tokenizer,
    text,
    span_start,
    span_end,
    device,
    require_single_token=False,
):
    token_indices, input_ids, offsets = get_changed_token_indices(
        tokenizer, text, span_start, span_end
    )

    if not token_indices:
        return {
            "ok": False,
            "reason": "no_token_overlaps_changed_span",
            "changed_token_indices": [],
        }

    if require_single_token and len(token_indices) != 1:
        return {
            "ok": False,
            "reason": "changed_span_not_single_token",
            "changed_token_indices": token_indices,
            "num_changed_tokens": len(token_indices),
        }

    if token_indices[0] == 0:
        return {
            "ok": False,
            "reason": "changed_token_is_first_token_no_context",
            "changed_token_indices": token_indices,
        }

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    outputs = model(input_tensor)
    logits = outputs.logits[0]  # [seq_len, vocab]

    nlls = []
    token_texts = []

    for idx in token_indices:
        # Causal LM: token idx is predicted by logits at idx - 1.
        log_probs = F.log_softmax(logits[idx - 1], dim=-1)
        token_id = input_ids[idx]
        nll = -log_probs[token_id].item()
        nlls.append(nll)

        s, e = offsets[idx]
        token_texts.append(text[s:e])

    return {
        "ok": True,
        "sum_nll": float(sum(nlls)),
        "mean_nll": float(sum(nlls) / len(nlls)),
        "token_nlls": nlls,
        "changed_token_indices": token_indices,
        "changed_token_texts": token_texts,
        "num_changed_tokens": len(token_indices),
    }


def score_row(model, tokenizer, row, device, require_single_token=False):
    prefix = row["prefix"]
    target = row["target"]
    substitute = row.get("substitute") or row.get("selected_candidate")
    suffix = row["suffix"]

    if substitute is None:
        return None, {
            **row,
            "score_ok": False,
            "score_reason": "missing_substitute",
        }

    orig_text = prefix + target + suffix
    pert_text = prefix + substitute + suffix

    orig_start = len(prefix)
    orig_end = len(prefix) + len(target)

    pert_start = len(prefix)
    pert_end = len(prefix) + len(substitute)

    orig_score = changed_span_nll(
        model=model,
        tokenizer=tokenizer,
        text=orig_text,
        span_start=orig_start,
        span_end=orig_end,
        device=device,
        require_single_token=require_single_token,
    )

    pert_score = changed_span_nll(
        model=model,
        tokenizer=tokenizer,
        text=pert_text,
        span_start=pert_start,
        span_end=pert_end,
        device=device,
        require_single_token=require_single_token,
    )

    if not orig_score["ok"] or not pert_score["ok"]:
        skipped = {
            **row,
            "score_ok": False,
            "orig_score": orig_score,
            "pert_score": pert_score,
        }
        return None, skipped

    out = {
        **row,
        "score_ok": True,

        "orig_text": orig_text,
        "pert_text": pert_text,

        "orig_changed_text": target,
        "pert_changed_text": substitute,

        "orig_changed_sum_nll": orig_score["sum_nll"],
        "pert_changed_sum_nll": pert_score["sum_nll"],
        "delta_sum_nll": pert_score["sum_nll"] - orig_score["sum_nll"],

        "orig_changed_mean_nll": orig_score["mean_nll"],
        "pert_changed_mean_nll": pert_score["mean_nll"],
        "delta_mean_nll": pert_score["mean_nll"] - orig_score["mean_nll"],

        "orig_num_changed_tokens": orig_score["num_changed_tokens"],
        "pert_num_changed_tokens": pert_score["num_changed_tokens"],
        "orig_changed_token_texts": orig_score["changed_token_texts"],
        "pert_changed_token_texts": pert_score["changed_token_texts"],
        "orig_token_nlls": orig_score["token_nlls"],
        "pert_token_nlls": pert_score["token_nlls"],
    }

    return out, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Manual selected JSONL")
    parser.add_argument("--output", required=True, help="Scored JSONL output")
    parser.add_argument("--skipped_output", default=None)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-7B",
        help="Base model for likelihood scoring",
    )
    parser.add_argument(
        "--require_single_token",
        action="store_true",
        help="Skip rows unless both original and substitute changed spans are exactly one tokenizer token.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    rows = load_jsonl(args.input)
    scored = []
    skipped = []

    for row in tqdm(rows, desc="Scoring changed word NLL"):
        out, skip = score_row(
            model=model,
            tokenizer=tokenizer,
            row=row,
            device=device,
            require_single_token=args.require_single_token,
        )
        if out is not None:
            scored.append(out)
        if skip is not None:
            skipped.append(skip)

    write_jsonl(args.output, scored)

    if args.skipped_output is None:
        args.skipped_output = args.output.replace(".jsonl", ".skipped.jsonl")
    write_jsonl(args.skipped_output, skipped)

    if scored:
        mean_delta_sum = sum(r["delta_sum_nll"] for r in scored) / len(scored)
        frac_positive = sum(r["delta_sum_nll"] > 0 for r in scored) / len(scored)
        positive = [r["delta_sum_nll"] for r in scored if r["delta_sum_nll"] > 0]
        positive_half_mean = sum(positive) / len(positive) if positive else 0.0
    else:
        mean_delta_sum = 0.0
        frac_positive = 0.0
        positive_half_mean = 0.0

    summary = {
        "input": args.input,
        "output": args.output,
        "model": args.model,
        "n_total": len(rows),
        "n_scored": len(scored),
        "n_skipped": len(skipped),
        "mean_delta_sum_nll": mean_delta_sum,
        "frac_positive_delta_sum_nll": frac_positive,
        "positive_half_mean_delta_sum_nll": positive_half_mean,
        "interpretation": "delta_sum_nll = perturbed_changed_word_nll - original_changed_word_nll; positive means model prefers original changed word.",
    }

    summary_path = args.output.replace(".jsonl", ".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()