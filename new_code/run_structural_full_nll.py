#!/usr/bin/env python3
"""
Score structural perturbation JSONL with whole-instance mean NLL.

Input JSONL schema expected:
  - source_code: original full code string
  - perturbed_source_code: transformed full code string
  - optional: example_id, dataset, transform_type, transform_id, etc.

Output JSONL adds:
  - orig_full_mean_nll
  - pert_full_mean_nll
  - delta_full_mean_nll = pert_full_mean_nll - orig_full_mean_nll
  - prefers_original_full

Important:
  This scores the whole code instance, not only the modified AST span.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_input_line_no"] = row.get("_input_line_no", line_no)
            yield row


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_max_context(model: Any, tokenizer: Any, user_max_length: Optional[int]) -> Optional[int]:
    if user_max_length is not None and user_max_length > 0:
        return user_max_length

    for attr in ("max_position_embeddings", "n_positions", "seq_length"):
        value = getattr(model.config, attr, None)
        if isinstance(value, int) and value > 0:
            return value

    value = getattr(tokenizer, "model_max_length", None)
    if isinstance(value, int) and 0 < value < 10**7:
        return value

    return None


def maybe_truncate_ids(
    input_ids: torch.Tensor,
    max_context: Optional[int],
    truncate: str,
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    n = int(input_ids.numel())

    if max_context is None or n <= max_context:
        return input_ids, None

    if truncate == "none":
        return None, f"too_long_{n}_tokens_gt_{max_context}"

    if truncate == "left":
        return input_ids[-max_context:], f"left_truncated_{n}_to_{max_context}"

    if truncate == "right":
        return input_ids[:max_context], f"right_truncated_{n}_to_{max_context}"

    raise ValueError(f"Unknown truncate mode: {truncate}")


@torch.inference_mode()
def full_text_mean_nll(
    text: str,
    *,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    max_context: Optional[int],
    truncate: str,
    add_special_tokens: bool,
    save_token_nlls: bool = False,
) -> Dict[str, Any]:
    """
    Compute average causal-LM NLL over the whole text.

    For tokens t_0 ... t_{n-1}, score t_1 ... t_{n-1}.
    The first token has no previous context, so it is not scored.
    """
    if not isinstance(text, str) or text == "":
        return {"scored": False, "skip_reason": "empty_text"}

    enc = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )

    ids = enc["input_ids"][0]
    total_tokens_before_trunc = int(ids.numel())

    ids, trunc_note = maybe_truncate_ids(
        ids,
        max_context=max_context,
        truncate=truncate,
    )

    if ids is None:
        return {
            "scored": False,
            "skip_reason": trunc_note,
            "num_total_tokens": total_tokens_before_trunc,
        }

    total_tokens = int(ids.numel())

    if total_tokens < 2:
        return {
            "scored": False,
            "skip_reason": "fewer_than_2_tokens",
            "num_total_tokens": total_tokens,
        }

    input_ids = ids.unsqueeze(0).to(device)

    outputs = model(input_ids=input_ids)
    logits = outputs.logits

    # logits[:, i, :] predicts input_ids[:, i + 1]
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()

    token_nlls = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )

    if torch.any(token_nlls < -1e-7):
        min_nll = float(token_nlls.min().detach().cpu())
        raise RuntimeError(f"Negative token NLL detected: {min_nll}")

    token_nlls = torch.clamp(token_nlls, min=0.0)

    result: Dict[str, Any] = {
        "scored": True,
        "mean_nll": float(token_nlls.mean().detach().cpu()),
        "sum_nll": float(token_nlls.sum().detach().cpu()),
        "num_score_tokens": int(token_nlls.numel()),
        "num_total_tokens": total_tokens,
        "num_total_tokens_before_trunc": total_tokens_before_trunc,
    }

    if trunc_note is not None:
        result["truncation_note"] = trunc_note

    if save_token_nlls:
        toks = tokenizer.convert_ids_to_tokens(ids.tolist())
        result["tokens"] = toks
        result["scored_tokens"] = toks[1:]
        result["token_nlls"] = [float(x) for x in token_nlls.detach().cpu().tolist()]

    return result


def summarize(rows: List[Dict[str, Any]], tie_eps: float) -> Dict[str, Any]:
    scored = [r for r in rows if r.get("scored") is True]
    skipped = [r for r in rows if r.get("scored") is not True]
    deltas = [float(r["delta_full_mean_nll"]) for r in scored]

    def stats(xs: List[float]) -> Dict[str, Any]:
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "mean": statistics.mean(xs),
            "median": statistics.median(xs),
            "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs),
            "max": max(xs),
            "frac_positive": sum(x > tie_eps for x in xs) / len(xs),
            "frac_negative": sum(x < -tie_eps for x in xs) / len(xs),
        }

    by_transform: Dict[str, List[float]] = defaultdict(list)

    for r in scored:
        transform_type = str(r.get("transform_type", "unknown"))
        by_transform[transform_type].append(float(r["delta_full_mean_nll"]))

    return {
        "n_total_rows": len(rows),
        "n_scored_rows": len(scored),
        "n_skipped_rows": len(skipped),
        "tie_eps": tie_eps,
        "delta_full_mean_nll": stats(deltas),
        "mean_orig_full_mean_nll": (
            statistics.mean([float(r["orig_full_mean_nll"]) for r in scored])
            if scored
            else None
        ),
        "mean_pert_full_mean_nll": (
            statistics.mean([float(r["pert_full_mean_nll"]) for r in scored])
            if scored
            else None
        ),
        "counts": {
            "prefers_original_full": sum(
                float(r["delta_full_mean_nll"]) > tie_eps for r in scored
            ),
            "prefers_perturbed_full": sum(
                float(r["delta_full_mean_nll"]) < -tie_eps for r in scored
            ),
            "ties_full": sum(
                abs(float(r["delta_full_mean_nll"])) <= tie_eps for r in scored
            ),
        },
        "counts_by_transform_type": dict(
            Counter(str(r.get("transform_type", "unknown")) for r in scored)
        ),
        "skip_reasons": dict(
            Counter(str(r.get("skip_reason", "unknown")) for r in skipped)
        ),
        "by_transform_type": {
            k: stats(v)
            for k, v in sorted(by_transform.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out_jsonl", required=True, type=Path)
    parser.add_argument("--out_summary", required=True, type=Path)

    parser.add_argument("--model", default="bigcode/starcoderbase-3b")
    parser.add_argument("--orig_field", default="source_code")
    parser.add_argument("--pert_field", default="perturbed_source_code")

    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument(
        "--truncate",
        choices=["none", "left", "right"],
        default="none",
    )

    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tie_eps", type=float, default=1e-8)

    parser.add_argument("--add_special_tokens", action="store_true")
    parser.add_argument("--save_token_nlls", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.dtype == "auto":
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()

    max_context = resolve_max_context(model, tokenizer, args.max_length)

    rows_in = list(read_jsonl(args.input))
    if args.max_rows is not None:
        rows_in = rows_in[: args.max_rows]

    rows_out: List[Dict[str, Any]] = []

    for row in tqdm(rows_in, desc="Scoring whole-instance NLL"):
        out = dict(row)

        orig_text = row.get(args.orig_field)
        pert_text = row.get(args.pert_field)

        if not isinstance(orig_text, str) or not isinstance(pert_text, str):
            out["scored"] = False
            out["skip_reason"] = "missing_source_or_perturbed_source"
            rows_out.append(out)
            continue

        orig = full_text_mean_nll(
            orig_text,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_context=max_context,
            truncate=args.truncate,
            add_special_tokens=args.add_special_tokens,
            save_token_nlls=args.save_token_nlls,
        )

        pert = full_text_mean_nll(
            pert_text,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_context=max_context,
            truncate=args.truncate,
            add_special_tokens=args.add_special_tokens,
            save_token_nlls=args.save_token_nlls,
        )

        if not orig.get("scored"):
            out["scored"] = False
            out["skip_reason"] = "orig_" + str(orig.get("skip_reason", "unknown"))
            rows_out.append(out)
            continue

        if not pert.get("scored"):
            out["scored"] = False
            out["skip_reason"] = "pert_" + str(pert.get("skip_reason", "unknown"))
            rows_out.append(out)
            continue

        orig_mean = float(orig["mean_nll"])
        pert_mean = float(pert["mean_nll"])
        delta = pert_mean - orig_mean

        out.update(
            {
                "scored": True,
                "orig_full_mean_nll": orig_mean,
                "pert_full_mean_nll": pert_mean,
                "delta_full_mean_nll": delta,
                "prefers_original_full": delta > args.tie_eps,
                "preference_label_full": (
                    "prefers_original"
                    if delta > args.tie_eps
                    else "prefers_perturbed"
                    if delta < -args.tie_eps
                    else "tie"
                ),
                "orig_full_sum_nll": float(orig["sum_nll"]),
                "pert_full_sum_nll": float(pert["sum_nll"]),
                "orig_full_num_score_tokens": int(orig["num_score_tokens"]),
                "pert_full_num_score_tokens": int(pert["num_score_tokens"]),
                "orig_full_num_total_tokens": int(orig["num_total_tokens"]),
                "pert_full_num_total_tokens": int(pert["num_total_tokens"]),
                "orig_full_num_total_tokens_before_trunc": int(
                    orig["num_total_tokens_before_trunc"]
                ),
                "pert_full_num_total_tokens_before_trunc": int(
                    pert["num_total_tokens_before_trunc"]
                ),
            }
        )

        if "truncation_note" in orig:
            out["orig_truncation_note"] = orig["truncation_note"]

        if "truncation_note" in pert:
            out["pert_truncation_note"] = pert["truncation_note"]

        if args.save_token_nlls:
            out["orig_full_tokens"] = orig.get("tokens")
            out["orig_full_scored_tokens"] = orig.get("scored_tokens")
            out["orig_full_token_nlls"] = orig.get("token_nlls")
            out["pert_full_tokens"] = pert.get("tokens")
            out["pert_full_scored_tokens"] = pert.get("scored_tokens")
            out["pert_full_token_nlls"] = pert.get("token_nlls")

        rows_out.append(out)

    write_jsonl(args.out_jsonl, rows_out)

    summary = summarize(rows_out, tie_eps=args.tie_eps)
    summary.update(
        {
            "input_file": str(args.input),
            "output_file": str(args.out_jsonl),
            "model": args.model,
            "orig_field": args.orig_field,
            "pert_field": args.pert_field,
            "device": str(device),
            "dtype": str(torch_dtype),
            "max_context": max_context,
            "truncate": args.truncate,
            "add_special_tokens": args.add_special_tokens,
            "metric_note": (
                "Whole-instance mean NLL. "
                "delta_full_mean_nll = pert_full_mean_nll - orig_full_mean_nll. "
                "Positive means the model prefers the original full code instance."
            ),
        }
    )

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()