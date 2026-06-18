#!/usr/bin/env python3
"""
Split valid variable-name score rows into:
  1. rows where the model prefers the original variable name
  2. rows where the model prefers the substitute variable name
  3. near-tie rows

Input:
  *_var_scores.jsonl

Output:
  <prefix>_prefers_original.jsonl
  <prefix>_prefers_substitute.jsonl
  <prefix>_ties.jsonl
  <prefix>_summary.json
  <prefix>_inspection.csv

Example:
  python split_preference_rows.py \
    --input bigcodebench_var_scores.jsonl \
    --out_prefix analysis/bigcodebench

  python split_preference_rows.py \
    --input quixbugs_var_scores.jsonl \
    --out_prefix analysis/quixbugs \
    --require_tests_pass
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(x: Any) -> Optional[float]:
    try:
        y = float(x)
        if math.isfinite(y):
            return y
        return None
    except Exception:
        return None


def mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else None


def median(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if math.isfinite(x)]
    return statistics.median(xs) if xs else None


def stdev(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if math.isfinite(x)]
    return statistics.stdev(xs) if len(xs) >= 2 else None


def keep_valid(row: Dict[str, Any], require_tests_pass: bool) -> bool:
    """
    Valid row = actually scored by the NLL script.

    If require_tests_pass=True, also require both original and perturbed tests
    to pass. Usually this is what you want for BigCodeBench.
    """
    if row.get("scored") is not True:
        return False

    if safe_float(row.get("delta_mean_nll")) is None:
        return False

    if safe_float(row.get("orig_mean_nll")) is None:
        return False

    if safe_float(row.get("sub_mean_nll")) is None:
        return False

    # Raw NLL should never be negative.
    if safe_float(row.get("orig_mean_nll")) < 0:
        return False

    if safe_float(row.get("sub_mean_nll")) < 0:
        return False

    if require_tests_pass:
        if row.get("original_test_passed") is not True:
            return False
        if row.get("perturbed_test_passed") is not True:
            return False

    return True


def classify_row(row: Dict[str, Any], tie_eps: float) -> str:
    delta = safe_float(row.get("delta_mean_nll"))
    assert delta is not None

    if delta > tie_eps:
        return "prefers_original"

    if delta < -tie_eps:
        return "prefers_substitute"

    return "tie"


def compact_for_csv(row: Dict[str, Any], label: str) -> Dict[str, Any]:
    """
    A compact row for manual inspection in spreadsheet form.
    """
    orig_tokens = row.get("orig_tokens")
    sub_tokens = row.get("sub_tokens")
    orig_nlls = row.get("orig_token_nlls")
    sub_nlls = row.get("sub_token_nlls")

    return {
        "label": label,
        "example_id": row.get("example_id"),
        "dataset": row.get("dataset"),
        "old_name": row.get("old_name"),
        "new_name": row.get("new_name"),
        "delta_mean_nll": row.get("delta_mean_nll"),
        "orig_mean_nll": row.get("orig_mean_nll"),
        "sub_mean_nll": row.get("sub_mean_nll"),
        "orig_num_tokens": row.get("orig_num_tokens"),
        "sub_num_tokens": row.get("sub_num_tokens"),
        "orig_tokens": " ".join(map(str, orig_tokens)) if isinstance(orig_tokens, list) else orig_tokens,
        "sub_tokens": " ".join(map(str, sub_tokens)) if isinstance(sub_tokens, list) else sub_tokens,
        "orig_token_nlls": " ".join(f"{float(x):.4f}" for x in orig_nlls) if isinstance(orig_nlls, list) else orig_nlls,
        "sub_token_nlls": " ".join(f"{float(x):.4f}" for x in sub_nlls) if isinstance(sub_nlls, list) else sub_nlls,
        "first_line": row.get("first_line"),
        "first_column": row.get("first_column"),
        "binding_context": row.get("binding_context"),
        "selection_reason": row.get("selection_reason"),
        "original_test_passed": row.get("original_test_passed"),
        "perturbed_test_passed": row.get("perturbed_test_passed"),
    }


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "label",
        "example_id",
        "dataset",
        "old_name",
        "new_name",
        "delta_mean_nll",
        "orig_mean_nll",
        "sub_mean_nll",
        "orig_num_tokens",
        "sub_num_tokens",
        "orig_tokens",
        "sub_tokens",
        "orig_token_nlls",
        "sub_token_nlls",
        "first_line",
        "first_column",
        "binding_context",
        "selection_reason",
        "original_test_passed",
        "perturbed_test_passed",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    deltas = [safe_float(r.get("delta_mean_nll")) for r in rows]
    deltas = [x for x in deltas if x is not None]

    abs_deltas = [abs(x) for x in deltas]

    orig_nlls = [safe_float(r.get("orig_mean_nll")) for r in rows]
    orig_nlls = [x for x in orig_nlls if x is not None]

    sub_nlls = [safe_float(r.get("sub_mean_nll")) for r in rows]
    sub_nlls = [x for x in sub_nlls if x is not None]

    same_token_count = [
        r for r in rows
        if r.get("orig_num_tokens") is not None
        and r.get("sub_num_tokens") is not None
        and r.get("orig_num_tokens") == r.get("sub_num_tokens")
    ]

    return {
        "n": len(rows),
        "mean_delta_mean_nll": mean(deltas),
        "median_delta_mean_nll": median(deltas),
        "std_delta_mean_nll": stdev(deltas),
        "mean_abs_delta_mean_nll": mean(abs_deltas),
        "median_abs_delta_mean_nll": median(abs_deltas),
        "mean_orig_mean_nll": mean(orig_nlls),
        "mean_sub_mean_nll": mean(sub_nlls),
        "n_same_token_count": len(same_token_count),
        "frac_same_token_count": len(same_token_count) / len(rows) if rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, help="Input score JSONL.")
    parser.add_argument(
        "--out_prefix",
        required=True,
        help="Output prefix, e.g. analysis/bigcodebench.",
    )
    parser.add_argument(
        "--require_tests_pass",
        action="store_true",
        help="Only keep rows where original and perturbed tests both passed.",
    )
    parser.add_argument(
        "--tie_eps",
        type=float,
        default=1e-9,
        help="Rows with abs(delta_mean_nll) <= tie_eps are treated as ties.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of strongest examples to include in summary.",
    )

    args = parser.parse_args()

    rows = read_jsonl(args.input)

    valid_rows = [r for r in rows if keep_valid(r, args.require_tests_pass)]

    prefers_original = []
    prefers_substitute = []
    ties = []

    for row in valid_rows:
        label = classify_row(row, args.tie_eps)
        row = dict(row)
        row["preference_label"] = label

        if label == "prefers_original":
            prefers_original.append(row)
        elif label == "prefers_substitute":
            prefers_substitute.append(row)
        else:
            ties.append(row)

    # Sort by strength.
    # For prefers_original, strongest means largest positive delta.
    prefers_original.sort(
        key=lambda r: safe_float(r.get("delta_mean_nll")) or 0.0,
        reverse=True,
    )

    # For prefers_substitute, strongest means most negative delta.
    prefers_substitute.sort(
        key=lambda r: safe_float(r.get("delta_mean_nll")) or 0.0,
    )

    ties.sort(
        key=lambda r: abs(safe_float(r.get("delta_mean_nll")) or 0.0),
    )

    out_prefix = Path(args.out_prefix)

    write_jsonl(f"{out_prefix}_prefers_original.jsonl", prefers_original)
    write_jsonl(f"{out_prefix}_prefers_substitute.jsonl", prefers_substitute)
    write_jsonl(f"{out_prefix}_ties.jsonl", ties)

    csv_rows = []
    for r in prefers_original:
        csv_rows.append(compact_for_csv(r, "prefers_original"))
    for r in prefers_substitute:
        csv_rows.append(compact_for_csv(r, "prefers_substitute"))
    for r in ties:
        csv_rows.append(compact_for_csv(r, "tie"))

    write_csv(f"{out_prefix}_inspection.csv", csv_rows)

    all_deltas = [safe_float(r.get("delta_mean_nll")) for r in valid_rows]
    all_deltas = [x for x in all_deltas if x is not None]

    summary = {
        "input": args.input,
        "n_total_rows": len(rows),
        "n_valid_rows": len(valid_rows),
        "require_tests_pass": args.require_tests_pass,
        "tie_eps": args.tie_eps,
        "counts": {
            "prefers_original": len(prefers_original),
            "prefers_substitute": len(prefers_substitute),
            "ties": len(ties),
        },
        "fractions": {
            "prefers_original": len(prefers_original) / len(valid_rows) if valid_rows else None,
            "prefers_substitute": len(prefers_substitute) / len(valid_rows) if valid_rows else None,
            "ties": len(ties) / len(valid_rows) if valid_rows else None,
        },
        "overall": summarize_group(valid_rows),
        "prefers_original": summarize_group(prefers_original),
        "prefers_substitute": summarize_group(prefers_substitute),
        "ties": summarize_group(ties),
        "strongest_prefers_original": [
            {
                "example_id": r.get("example_id"),
                "old_name": r.get("old_name"),
                "new_name": r.get("new_name"),
                "delta_mean_nll": r.get("delta_mean_nll"),
                "orig_tokens": r.get("orig_tokens"),
                "sub_tokens": r.get("sub_tokens"),
                "binding_context": r.get("binding_context"),
            }
            for r in prefers_original[: args.top_k]
        ],
        "strongest_prefers_substitute": [
            {
                "example_id": r.get("example_id"),
                "old_name": r.get("old_name"),
                "new_name": r.get("new_name"),
                "delta_mean_nll": r.get("delta_mean_nll"),
                "orig_tokens": r.get("orig_tokens"),
                "sub_tokens": r.get("sub_tokens"),
                "binding_context": r.get("binding_context"),
            }
            for r in prefers_substitute[: args.top_k]
        ],
    }

    summary_path = f"{out_prefix}_summary.json"
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary["counts"], indent=2, ensure_ascii=False))
    print(json.dumps(summary["fractions"], indent=2, ensure_ascii=False))
    print()
    print(f"Wrote: {out_prefix}_prefers_original.jsonl")
    print(f"Wrote: {out_prefix}_prefers_substitute.jsonl")
    print(f"Wrote: {out_prefix}_ties.jsonl")
    print(f"Wrote: {out_prefix}_inspection.csv")
    print(f"Wrote: {out_prefix}_summary.json")


if __name__ == "__main__":
    main()