#!/usr/bin/env python3
"""
Generate semantics-preserving structural perturbations from an existing JSONL file.

Input rows may come from the old variable-name pipeline. This script ignores all
variable-name fields and uses only:
  - example_id
  - dataset
  - source_code
  - optional test / entry_point fields, if present

Supported structural transforms, intentionally conservative:
  1. comparison_mirror:
       a < b   ->   b > a
       a <= b  ->   b >= a
       a > b   ->   b < a
       a >= b  ->   b <= a
       a == b  ->   b == a
       a != b  ->   b != a

  2. if_else_inversion:
       if cond:
           A
       else:
           B
       ->
       if not (cond):
           B
       else:
           A

No for-loop-to-while-loop transform is included.

Output rows are compatible with a continuation-style NLL scorer:
  prefix = source_code[:first_start_char]
  original_continuation = original structural span
  substitute_continuation = transformed structural span

Example:
  python generate_structural_perturbations.py \
    --input quixbugs_var_scores.jsonl \
    --out_jsonl quixbugs_structural_candidates.jsonl \
    --out_summary quixbugs_structural_summary.json \
    --max_per_example 1
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -------------------------
# JSON helpers
# -------------------------

def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_input_line_no"] = line_no
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# -------------------------
# Source position helpers
# -------------------------

def line_offsets(src: str) -> List[int]:
    offsets = [0]
    total = 0
    for line in src.splitlines(keepends=True):
        total += len(line)
        offsets.append(total)
    return offsets


def node_span(src: str, node: ast.AST) -> Tuple[int, int]:
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        raise ValueError("AST node is missing lineno/end_lineno; use Python 3.8+.")
    offs = line_offsets(src)
    start = offs[node.lineno - 1] + node.col_offset
    end = offs[node.end_lineno - 1] + node.end_col_offset
    return start, end


def get_text(src: str, node: ast.AST) -> str:
    start, end = node_span(src, node)
    return src[start:end]


def line_text(src: str, lineno: int) -> str:
    lines = src.splitlines(keepends=True)
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1]
    return ""


def leading_indent(s: str) -> str:
    return s[: len(s) - len(s.lstrip(" \t"))]


# -------------------------
# Safety checks
# -------------------------

def is_simple_pure_expr(node: ast.AST) -> bool:
    """
    Conservative side-effect filter for comparison mirroring.

    We allow common expression forms that are usually read-only and avoid calls,
    comprehensions, assignments, yields, awaits, etc.
    """
    if isinstance(node, (ast.Name, ast.Constant)):
        return True

    if isinstance(node, ast.Attribute):
        return is_simple_pure_expr(node.value)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
        return is_simple_pure_expr(node.operand)

    if isinstance(node, ast.Subscript):
        return is_simple_pure_expr(node.value) and is_simple_pure_expr(node.slice)

    if isinstance(node, ast.Tuple):
        return all(is_simple_pure_expr(elt) for elt in node.elts)

    if isinstance(node, ast.List):
        return all(is_simple_pure_expr(elt) for elt in node.elts)

    # Python 3.9+: slices are expressions directly. Older versions had ast.Index.
    if hasattr(ast, "Index") and isinstance(node, ast.Index):  # type: ignore[attr-defined]
        return is_simple_pure_expr(node.value)  # type: ignore[attr-defined]

    if isinstance(node, ast.Slice):
        parts = [node.lower, node.upper, node.step]
        return all(part is None or is_simple_pure_expr(part) for part in parts)

    return False


# -------------------------
# Transform generation
# -------------------------

OP_MIRROR = {
    ast.Lt: ">",
    ast.LtE: ">=",
    ast.Gt: "<",
    ast.GtE: "<=",
    ast.Eq: "==",
    ast.NotEq: "!=",
}


def make_perturbed_source(src: str, start: int, end: int, replacement: str) -> str:
    return src[:start] + replacement + src[end:]


def comparison_mirror_candidates(src: str, tree: ast.AST) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if len(node.ops) != 1 or len(node.comparators) != 1:
            continue

        op = node.ops[0]
        op_type = type(op)
        if op_type not in OP_MIRROR:
            continue

        left = node.left
        right = node.comparators[0]

        if not (is_simple_pure_expr(left) and is_simple_pure_expr(right)):
            continue

        start, end = node_span(src, node)
        left_src = get_text(src, left)
        right_src = get_text(src, right)
        original = src[start:end]
        replacement = f"{right_src} {OP_MIRROR[op_type]} {left_src}"

        if replacement == original:
            continue

        candidates.append({
            "transform_type": "comparison_mirror",
            "first_start_char": start,
            "first_end_char": end,
            "first_line": node.lineno,
            "first_column": node.col_offset,
            "original_continuation": original,
            "substitute_continuation": replacement,
            "transform_reason": "Mirror a simple comparison by swapping operands and reversing the comparison operator.",
        })

    return candidates


def stmt_block_text(src: str, stmts: List[ast.stmt]) -> Optional[str]:
    if not stmts:
        return None

    # For a statement block we need to preserve the indentation before the
    # first statement. AST col_offset points to the first token, so use the
    # beginning of the physical line instead.
    offs = line_offsets(src)
    start = offs[stmts[0].lineno - 1]
    _, end = node_span(src, stmts[-1])
    return src[start:end]


def if_else_inversion_candidates(src: str, tree: ast.AST) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not node.body or not node.orelse:
            continue

        # Skip elif chains. In AST, elif is represented as orelse=[If(...)] whose
        # source line starts with "elif".
        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            if line_text(src, node.orelse[0].lineno).lstrip().startswith("elif"):
                continue

        test_src = get_text(src, node.test)
        body_text = stmt_block_text(src, node.body)
        else_text = stmt_block_text(src, node.orelse)
        if body_text is None or else_text is None:
            continue

        start, end = node_span(src, node)
        original = src[start:end]
        indent = leading_indent(line_text(src, node.lineno))

        # The replacement span starts at the `if` token, not at the beginning
        # of the line, so the first line must not include leading indentation.
        # Subsequent lines need their original indentation.
        replacement = (
            f"if not ({test_src}):\n"
            f"{else_text}\n"
            f"{indent}else:\n"
            f"{body_text}"
        )

        if replacement == original:
            continue

        candidates.append({
            "transform_type": "if_else_inversion",
            "first_start_char": start,
            "first_end_char": end,
            "first_line": node.lineno,
            "first_column": node.col_offset,
            "original_continuation": original,
            "substitute_continuation": replacement,
            "transform_reason": "Invert an if/else condition and swap the two branches.",
        })

    return candidates


def generate_candidates(src: str) -> List[Dict[str, Any]]:
    tree = ast.parse(src)
    candidates = []
    candidates.extend(comparison_mirror_candidates(src, tree))
    candidates.extend(if_else_inversion_candidates(src, tree))

    # Stable order: earlier source span first, then smaller local replacement first.
    candidates.sort(key=lambda c: (
        c["first_start_char"],
        c["first_end_char"] - c["first_start_char"],
        c["transform_type"],
    ))

    for i, c in enumerate(candidates):
        c["transform_id"] = f"{c['transform_type']}@{c['first_start_char']}:{c['first_end_char']}#{i}"
    return candidates


# -------------------------
# Row conversion
# -------------------------

VARIABLE_RENAME_FIELDS = {
    "old_name", "new_name", "selected_candidate_id",
    "binding_context", "selection_reason",
    "renamed_occurrences", "renamed_references",
    "original_continuation", "substitute_continuation",
    "orig_mean_nll", "sub_mean_nll", "delta_mean_nll",
    "orig_num_tokens", "sub_num_tokens",
    "orig_tokens", "sub_tokens",
    "orig_token_ids", "sub_token_ids",
    "orig_token_nlls", "sub_token_nlls",
    "orig_token_offsets", "sub_token_offsets",
    "prefers_original", "preference_label",
}


def structural_row_from_candidate(row: Dict[str, Any], cand: Dict[str, Any]) -> Dict[str, Any]:
    src = row["source_code"]
    start = cand["first_start_char"]
    end = cand["first_end_char"]
    replacement = cand["substitute_continuation"]

    out: Dict[str, Any] = {}

    # Preserve useful non-variable metadata and test fields if they exist.
    for k, v in row.items():
        if k in VARIABLE_RENAME_FIELDS:
            continue
        if k in {"prefix", "perturbed_source_code", "scored", "decision", "first_start_char", "first_end_char", "first_line", "first_column"}:
            continue
        out[k] = v

    out.update({
        "decision": "ACCEPT",
        "transform_id": cand["transform_id"],
        "transform_type": cand["transform_type"],
        "first_start_char": start,
        "first_end_char": end,
        "first_line": cand["first_line"],
        "first_column": cand["first_column"],
        "prefix": src[:start],
        "original_continuation": cand["original_continuation"],
        "substitute_continuation": replacement,
        "transform_reason": cand["transform_reason"],
        "source_code": src,
        "perturbed_source_code": make_perturbed_source(src, start, end, replacement),
    })
    return out


def reject_row(row: Dict[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "example_id": row.get("example_id"),
        "dataset": row.get("dataset"),
        "decision": "REJECT",
        "skip_reason": reason,
        "source_code": row.get("source_code"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--out_summary", required=True)
    parser.add_argument("--max_per_example", type=int, default=1)
    parser.add_argument(
        "--include_rejects",
        action="store_true",
        help="Include REJECT rows for examples with no safe structural transform.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    outputs: List[Dict[str, Any]] = []

    counts_by_transform: Dict[str, int] = {}
    no_source = 0
    no_candidate = 0
    parse_failed = 0

    for row in rows:
        src = row.get("source_code")
        if not isinstance(src, str) or not src.strip():
            no_source += 1
            if args.include_rejects:
                outputs.append(reject_row(row, "missing_source_code"))
            continue

        try:
            candidates = generate_candidates(src)
        except SyntaxError as e:
            parse_failed += 1
            if args.include_rejects:
                r = reject_row(row, "parse_failed")
                r["error"] = str(e)
                outputs.append(r)
            continue

        if not candidates:
            no_candidate += 1
            if args.include_rejects:
                outputs.append(reject_row(row, "no_safe_structural_transform"))
            continue

        for cand in candidates[: args.max_per_example]:
            outputs.append(structural_row_from_candidate(row, cand))
            counts_by_transform[cand["transform_type"]] = counts_by_transform.get(cand["transform_type"], 0) + 1

    summary = {
        "input": args.input,
        "n_input_rows": len(rows),
        "n_output_rows": len(outputs),
        "max_per_example": args.max_per_example,
        "n_no_source": no_source,
        "n_parse_failed": parse_failed,
        "n_no_safe_structural_transform": no_candidate,
        "counts_by_transform": counts_by_transform,
    }

    write_jsonl(args.out_jsonl, outputs)
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {args.out_summary}")


if __name__ == "__main__":
    main()
