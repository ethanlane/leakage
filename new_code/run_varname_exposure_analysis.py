#!/usr/bin/env python3
"""
Run first-binding variable-name exposure analysis.

This version scores ONLY the changed variable name at its first binding position:

    orig_mean_nll = mean token NLL(old_name | prefix)
    sub_mean_nll  = mean token NLL(new_name | prefix)
    delta_mean_nll = sub_mean_nll - orig_mean_nll

It does NOT score the rest of the code.

Positive delta_mean_nll means:
    the substitute has higher average NLL,
    so the model prefers the original variable name.

Install:
    pip install torch transformers libcst tqdm

Example:
    python run_varname_exposure_analysis.py \
      --input quixbugs_var_selected.jsonl \
      --dataset quixbugs \
      --model bigcode/starcoderbase-3b \
      --out_jsonl quixbugs_var_scores.jsonl \
      --out_summary quixbugs_var_summary.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import statistics
import sys
import traceback
import unittest
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import libcst as cst
from libcst.metadata import (
    Assignment,
    ComprehensionScope,
    FunctionScope,
    MetadataWrapper,
    PositionProvider,
    ScopeProvider,
)
from tqdm import tqdm


LOCAL_SCOPES = (FunctionScope, ComprehensionScope)


# -----------------------------
# JSONL helpers
# -----------------------------

def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_input_path"] = str(path)
            row["_line_no"] = line_no
            yield row


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# -----------------------------
# Position helpers
# -----------------------------

def linecol_to_offset(src: str, line: int, column: int) -> int:
    """
    libcst line is 1-indexed, column is 0-indexed.
    """
    lines = src.splitlines(keepends=True)
    return sum(len(lines[i]) for i in range(line - 1)) + column


def get_node_offsets(src: str, pos) -> Tuple[int, int]:
    start = linecol_to_offset(src, pos.start.line, pos.start.column)
    end = linecol_to_offset(src, pos.end.line, pos.end.column)
    return start, end


# -----------------------------
# Scope-aware selected-binding renaming
# -----------------------------

class _BindingRenamer(cst.CSTTransformer):
    def __init__(self, rename_ids: set[int], new_name: str):
        self.rename_ids = rename_ids
        self.new_name = new_name

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name):
        if id(original_node) in self.rename_ids:
            return updated_node.with_changes(value=self.new_name)
        return updated_node


def rename_selected_binding(
    src: str,
    old_name: str,
    new_name: str,
    first_start_char: int,
    first_end_char: int,
) -> Tuple[str, Dict[str, Any]]:
    """
    Rename exactly the selected local binding and all references to that binding.
    """
    if src[first_start_char:first_end_char] != old_name:
        raise ValueError(
            f"Position mismatch: source[{first_start_char}:{first_end_char}]="
            f"{src[first_start_char:first_end_char]!r}, expected {old_name!r}"
        )

    module = cst.parse_module(src)
    wrapper = MetadataWrapper(module)
    scope_map = wrapper.resolve(ScopeProvider)
    pos_map = wrapper.resolve(PositionProvider)

    target_assignment: Optional[Assignment] = None
    target_name_node: Optional[cst.Name] = None

    seen_scopes = set()

    for _, scope in scope_map.items():
        if id(scope) in seen_scopes:
            continue
        seen_scopes.add(id(scope))

        if not isinstance(scope, LOCAL_SCOPES):
            continue

        for assignment in scope.assignments:
            if not isinstance(assignment, Assignment):
                continue
            if assignment.name != old_name:
                continue

            node = assignment.node

            # Only local variable / loop target / comprehension target.
            # Skip parameters, imports, FunctionDef, ClassDef, etc.
            if not isinstance(node, cst.Name):
                continue

            start, end = get_node_offsets(src, pos_map[node])
            if start == first_start_char and end == first_end_char:
                target_assignment = assignment
                target_name_node = node
                break

        if target_assignment is not None:
            break

    if target_assignment is None or target_name_node is None:
        raise ValueError(
            f"Could not find selected binding for {old_name!r} at "
            f"{first_start_char}:{first_end_char}."
        )

    rename_ids = {id(target_name_node)}
    ref_count = 0

    for ref in target_assignment.references:
        if isinstance(ref.node, cst.Name):
            rename_ids.add(id(ref.node))
            ref_count += 1

    perturbed = wrapper.visit(_BindingRenamer(rename_ids, new_name)).code

    meta = {
        "renamed_occurrences": len(rename_ids),
        "renamed_references": ref_count,
    }
    return perturbed, meta


# -----------------------------
# Optional test execution gate
# -----------------------------

def _run_unittest_classes(ns: Dict[str, Any]) -> Tuple[bool, str]:
    classes = []

    for value in ns.values():
        try:
            if (
                isinstance(value, type)
                and issubclass(value, unittest.TestCase)
                and value is not unittest.TestCase
            ):
                classes.append(value)
        except TypeError:
            pass

    if not classes:
        return True, "no unittest.TestCase classes found"

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for cls in classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=0)
    result = runner.run(suite)

    return result.wasSuccessful(), stream.getvalue()


def run_tests_for_row(
    src: str,
    test_code: Optional[str],
    entry_point: Optional[str],
) -> Tuple[Optional[bool], str]:
    """
    Generic test runner.

    Supports:
      - HumanEval-style check(candidate)
      - unittest.TestCase classes
      - top-level assert tests

    Returns:
      (None, reason) if there is no test field.
      (True/False, detail) otherwise.
    """
    if not test_code or not str(test_code).strip():
        return None, "no test field"

    ns: Dict[str, Any] = {
        "__name__": "__analysis_test__",
        "__file__": "<analysis_test>",
    }

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(src, ns)
            exec(test_code, ns)

            # HumanEval-style tests.
            if "check" in ns and callable(ns["check"]):
                if not entry_point:
                    return False, "check(candidate) exists but entry_point is missing"
                if entry_point not in ns:
                    return False, f"entry_point {entry_point!r} not found"
                ns["check"](ns[entry_point])
                return True, "check(candidate) passed"

            # BigCodeBench-style unittest tests.
            ok, detail = _run_unittest_classes(ns)
            return ok, detail

    except Exception as e:
        return False, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"


# -----------------------------
# LM scoring
# -----------------------------

def continuation_nll(
    model,
    tokenizer,
    prefix: str,
    continuation: str,
    device: str,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute mean NLL of ONLY the variable-name continuation:

        mean NLL(continuation | prefix)

    This does NOT score the rest of the program.

    Implementation:
    - tokenize prefix + continuation with offset mapping
    - identify tokens overlapping the continuation character span
    - score only those target tokens
    - return per-token mean NLL

    This avoids the invalid negative raw-NLL issue caused by:
        NLL(prefix + continuation) - NLL(prefix)
    """
    import torch

    full_text = prefix + continuation
    cont_start = len(prefix)
    cont_end = len(full_text)

    enc = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    input_ids_list: List[int] = enc["input_ids"]
    offsets: List[Tuple[int, int]] = enc["offset_mapping"]

    if not input_ids_list:
        raise ValueError("Empty tokenization.")

    # Score only tokens overlapping the continuation span.
    score_token_indices = [
        i for i, (s, e) in enumerate(offsets)
        if e > cont_start and s < cont_end
    ]

    if not score_token_indices:
        raise ValueError(f"No tokens overlap continuation: {continuation!r}")

    # Optional left truncation, but never truncate away the variable-name tokens.
    if max_length is not None and len(input_ids_list) > max_length:
        keep_from = len(input_ids_list) - max_length

        if min(score_token_indices) < keep_from:
            raise ValueError(
                "max_length truncation would remove part of the continuation. "
                f"min_score_index={min(score_token_indices)}, keep_from={keep_from}"
            )

        input_ids_list = input_ids_list[keep_from:]
        offsets = offsets[keep_from:]
        score_token_indices = [i - keep_from for i in score_token_indices]

    if min(score_token_indices) == 0:
        raise ValueError(
            "Cannot score the first token because it has no previous context token. "
            "Use a longer prefix or avoid excessive truncation."
        )

    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    token_nlls = []
    token_logprobs = []
    token_ids = []
    token_strs = []
    token_offsets = []

    for token_pos in score_token_indices:
        target_id = int(input_ids[0, token_pos].item())

        # Causal LM: logits at token_pos - 1 predict token at token_pos.
        pred_logits = logits[0, token_pos - 1, :]
        log_probs = torch.log_softmax(pred_logits, dim=-1)

        logprob = log_probs[target_id]
        nll = -logprob

        token_ids.append(target_id)
        token_strs.append(tokenizer.convert_ids_to_tokens([target_id])[0])
        token_offsets.append(offsets[token_pos])
        token_logprobs.append(float(logprob.item()))
        token_nlls.append(float(nll.item()))

    sum_nll = float(sum(token_nlls))
    mean_nll = sum_nll / len(token_nlls)

    return {
        "mean_nll": mean_nll,
        "num_tokens": len(token_nlls),
        "token_ids": token_ids,
        "tokens": token_strs,
        "token_offsets": token_offsets,
        "token_nlls": token_nlls,
        "token_logprobs": token_logprobs,
    }


def load_model_and_tokenizer(model_name: str, device_arg: str, dtype_arg: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg

    if dtype_arg == "float16":
        dtype = torch.float16
    elif dtype_arg == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_arg == "float32":
        dtype = torch.float32
    elif dtype_arg == "auto":
        dtype = None
    else:
        raise ValueError(f"Unknown dtype: {dtype_arg}")

    # Avoid CPU float16 issues.
    if device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    kwargs = {"trust_remote_code": True}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.to(device)
    model.eval()

    return model, tokenizer, device


# -----------------------------
# Summary stats
# -----------------------------

def _finite(xs: List[float]) -> List[float]:
    return [x for x in xs if x is not None and math.isfinite(x)]


def mean(xs: List[float]) -> Optional[float]:
    xs = _finite(xs)
    return sum(xs) / len(xs) if xs else None


def median(xs: List[float]) -> Optional[float]:
    xs = _finite(xs)
    return statistics.median(xs) if xs else None


def stdev(xs: List[float]) -> Optional[float]:
    xs = _finite(xs)
    return statistics.stdev(xs) if len(xs) >= 2 else None


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    kept = [r for r in rows if r.get("scored")]

    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for r in kept:
        by_dataset.setdefault(r.get("dataset", "unknown"), []).append(r)

    def block(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        dm = [r["delta_mean_nll"] for r in rs]

        return {
            "n": len(rs),
            "mean_delta_mean_nll": mean(dm),
            "median_delta_mean_nll": median(dm),
            "std_delta_mean_nll": stdev(dm),
            "min_delta_mean_nll": min(_finite(dm)) if _finite(dm) else None,
            "max_delta_mean_nll": max(_finite(dm)) if _finite(dm) else None,
            "frac_positive_delta_mean_nll": mean(
                [1.0 if x > 0 else 0.0 for x in _finite(dm)]
            ),
            "frac_prefers_original": mean(
                [1.0 if r.get("prefers_original") else 0.0 for r in rs]
            ),
        }

    return {
        "n_rows": len(rows),
        "n_scored": len(kept),
        "n_accept": sum(1 for r in rows if r.get("decision") == "ACCEPT"),
        "n_reject": sum(1 for r in rows if r.get("decision") == "REJECT"),
        "n_position_mismatch": sum(
            1 for r in rows if r.get("skip_reason") == "position_mismatch"
        ),
        "n_rename_failed": sum(
            1 for r in rows if r.get("skip_reason") == "rename_failed"
        ),
        "n_scoring_failed": sum(
            1 for r in rows if r.get("skip_reason") == "scoring_failed"
        ),
        "n_original_test_failed": sum(
            1 for r in rows if r.get("original_test_passed") is False
        ),
        "n_perturbed_test_failed": sum(
            1 for r in rows if r.get("perturbed_test_passed") is False
        ),
        "overall": block(kept),
        "by_dataset": {k: block(v) for k, v in sorted(by_dataset.items())},
    }


# -----------------------------
# Main row processing
# -----------------------------

def infer_dataset(row: Dict[str, Any], default_dataset: Optional[str]) -> str:
    if default_dataset:
        return default_dataset

    if row.get("dataset"):
        return str(row["dataset"])

    example_id = str(row.get("example_id", ""))
    if example_id.startswith("BigCodeBench"):
        return "bigcodebench"

    path = Path(row.get("_input_path", "unknown"))
    stem = path.stem.lower()

    if "quix" in stem:
        return "quixbugs"
    if "bigcode" in stem:
        return "bigcodebench"

    return stem or "unknown"


def process_row(
    row: Dict[str, Any],
    model,
    tokenizer,
    device: str,
    dataset: Optional[str],
    run_tests: bool,
    require_tests_pass: bool,
    max_length: Optional[int],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "example_id": row.get("example_id"),
        "dataset": infer_dataset(row, dataset),
        "decision": row.get("decision"),
        "old_name": row.get("old_name"),
        "new_name": row.get("new_name"),
        "selected_candidate_id": row.get("selected_candidate_id"),
        "first_start_char": row.get("first_start_char"),
        "first_end_char": row.get("first_end_char"),
        "first_line": row.get("first_line"),
        "first_column": row.get("first_column"),
        "binding_context": row.get("binding_context"),
        "selection_reason": row.get("selection_reason"),
        "source_code": row.get("source_code"),
    }

    if row.get("decision") != "ACCEPT":
        out["scored"] = False
        out["skip_reason"] = "not_accept"
        return out

    src = row.get("source_code")
    old = row.get("old_name")
    new = row.get("new_name")
    start = row.get("first_start_char")
    end = row.get("first_end_char")

    if not isinstance(src, str) or not isinstance(old, str) or not isinstance(new, str):
        out["scored"] = False
        out["skip_reason"] = "missing_required_fields"
        return out

    if not isinstance(start, int) or not isinstance(end, int):
        out["scored"] = False
        out["skip_reason"] = "missing_position"
        return out

    if src[start:end] != old:
        out["scored"] = False
        out["skip_reason"] = "position_mismatch"
        out["position_text"] = src[start:end]
        return out

    prefix = src[:start]

    out["prefix"] = prefix
    out["original_continuation"] = old
    out["substitute_continuation"] = new

    try:
        perturbed, rename_meta = rename_selected_binding(src, old, new, start, end)
        out.update(rename_meta)
        out["perturbed_source_code"] = perturbed
    except Exception as e:
        out["scored"] = False
        out["skip_reason"] = "rename_failed"
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    if run_tests:
        original_ok, original_detail = run_tests_for_row(
            src,
            row.get("test"),
            row.get("entry_point"),
        )
        perturbed_ok, perturbed_detail = run_tests_for_row(
            perturbed,
            row.get("test"),
            row.get("entry_point"),
        )

        out["original_test_passed"] = original_ok
        out["perturbed_test_passed"] = perturbed_ok
        out["original_test_detail"] = str(original_detail)[:2000]
        out["perturbed_test_detail"] = str(perturbed_detail)[:2000]

        if require_tests_pass and not (original_ok is True and perturbed_ok is True):
            out["scored"] = False
            out["skip_reason"] = "test_failed_or_missing"
            return out
    else:
        out["original_test_passed"] = None
        out["perturbed_test_passed"] = None

    try:
        orig_score = continuation_nll(
            model=model,
            tokenizer=tokenizer,
            prefix=prefix,
            continuation=old,
            device=device,
            max_length=max_length,
        )
        sub_score = continuation_nll(
            model=model,
            tokenizer=tokenizer,
            prefix=prefix,
            continuation=new,
            device=device,
            max_length=max_length,
        )
    except Exception as e:
        out["scored"] = False
        out["skip_reason"] = "scoring_failed"
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    out["orig_mean_nll"] = orig_score["mean_nll"]
    out["sub_mean_nll"] = sub_score["mean_nll"]
    out["delta_mean_nll"] = sub_score["mean_nll"] - orig_score["mean_nll"]

    out["orig_num_tokens"] = orig_score["num_tokens"]
    out["sub_num_tokens"] = sub_score["num_tokens"]

    out["orig_tokens"] = orig_score["tokens"]
    out["sub_tokens"] = sub_score["tokens"]
    out["orig_token_ids"] = orig_score["token_ids"]
    out["sub_token_ids"] = sub_score["token_ids"]
    out["orig_token_nlls"] = orig_score["token_nlls"]
    out["sub_token_nlls"] = sub_score["token_nlls"]
    out["orig_token_offsets"] = orig_score["token_offsets"]
    out["sub_token_offsets"] = sub_score["token_offsets"]

    out["scored"] = True
    out["prefers_original"] = out["delta_mean_nll"] > 0

    return out


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Claude-selected JSONL. Can be repeated.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset label, e.g. quixbugs or bigcodebench.",
    )
    parser.add_argument(
        "--model",
        default="bigcode/starcoderbase-3b",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32", "auto"],
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Optional left truncation length. Prefer None unless OOM.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--run_tests",
        action="store_true",
        help="Run original and perturbed benchmark tests.",
    )
    parser.add_argument(
        "--require_tests_pass",
        action="store_true",
        help="Only score rows where both original and perturbed tests pass.",
    )

    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--out_summary", required=True)

    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for path in args.input:
        rows.extend(list(read_jsonl(path)))

    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    print(f"Loaded {len(rows)} rows.", file=sys.stderr)
    print(f"Loading model: {args.model}", file=sys.stderr)

    model, tokenizer, device = load_model_and_tokenizer(
        model_name=args.model,
        device_arg=args.device,
        dtype_arg=args.dtype,
    )

    print(f"Using device: {device}", file=sys.stderr)

    results: List[Dict[str, Any]] = []

    for row in tqdm(rows, desc="scoring"):
        results.append(
            process_row(
                row=row,
                model=model,
                tokenizer=tokenizer,
                device=device,
                dataset=args.dataset,
                run_tests=args.run_tests,
                require_tests_pass=args.require_tests_pass,
                max_length=args.max_length,
            )
        )

    out_jsonl = Path(args.out_jsonl)
    out_summary = Path(args.out_summary)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(out_jsonl, results)

    summary = summarize(results)

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False), file=sys.stderr)
    print(f"Wrote results: {out_jsonl}", file=sys.stderr)
    print(f"Wrote summary: {out_summary}", file=sys.stderr)


if __name__ == "__main__":
    main()