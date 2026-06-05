#!/usr/bin/env python3
"""
Generate candidate pools for semantics-preserving local variable renaming.

Output: JSONL. Each row is one code example with a list of renamable candidates.
The LLM/Claude should pick exactly ONE candidate from each row.

Install:
    pip install libcst datasets tqdm

Example for BigCodeBench:
    python generate_var_candidate_pool.py \
        --hf_dataset bigcode/bigcodebench \
        --split v0.1.4 \
        --prefix_field code_prompt \
        --code_field canonical_solution \
        --id_field task_id \
        --out bigcodebench_var_candidates.jsonl

Example for a local JSONL file:
    python generate_var_candidate_pool.py \
        --input quixbugs.jsonl \
        --code_field code \
        --id_field name \
        --out quixbugs_var_candidates.jsonl
"""

from __future__ import annotations

import argparse
import builtins
import json
import keyword
import re
from dataclasses import asdict, dataclass
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

BAD_NAMES = set(keyword.kwlist) | set(dir(builtins)) | {
    "self",
    "cls",
    "_",
}

GENERIC_POOL = [
    "tmp", "val", "cur", "res", "out", "acc", "buf", "obj",
    "item", "elem", "node", "seq", "arr", "idx", "pos", "cnt",
    "total", "flag", "key", "value",
]

ROLE_POOLS = {
    "index": ["idx", "pos", "k", "p"],
    "counter": ["cnt", "total", "num"],
    "accumulator": ["acc", "res", "out", "total"],
    "result": ["res", "out", "ans"],
    "element": ["item", "elem", "val", "cur"],
    "sequence": ["seq", "arr", "items", "vals"],
    "mapping": ["mp", "table", "lookup"],
    "boolean": ["flag", "ok", "valid"],
    "temporary": ["tmp", "cur", "val"],
    "unknown": GENERIC_POOL,
}


@dataclass
class RenameCandidate:
    candidate_id: str

    old_name: str
    role: str
    occurrence_count: int

    first_start_char: int
    first_end_char: int
    first_line: int
    first_column: int
    binding_context: str

    candidate_new_names: List[str]

    # Useful for later scoring. The LLM does not need to reason over this.
    original_continuation: str


def linecol_to_offset(src: str, line: int, column: int) -> int:
    """
    libcst line is 1-indexed, column is 0-indexed.
    """
    lines = src.splitlines(keepends=True)
    return sum(len(lines[i]) for i in range(line - 1)) + column


def all_identifiers(src: str) -> set[str]:
    return set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", src))


def unique_keep_order(xs: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def is_clean_identifier(name: str) -> bool:
    if not name:
        return False
    if name in BAD_NAMES:
        return False
    if keyword.iskeyword(name):
        return False
    if not name.isidentifier():
        return False
    if name.startswith("_"):
        return False
    if name.isupper():
        return False
    return True


def infer_role(old_name: str, binding_context: str) -> str:
    n = old_name.lower()
    ctx = binding_context.lower()

    if n in {"i", "j", "k", "idx", "index", "pos", "position"}:
        return "index"
    if "range(" in ctx or "enumerate(" in ctx:
        return "index"

    if n in {"count", "counter", "cnt", "num", "n"}:
        return "counter"

    if n in {"acc", "accum", "sum", "total"}:
        return "accumulator"

    if n in {"result", "results", "res", "answer", "ans", "output", "out"}:
        return "result"

    if n in {"item", "elem", "element", "value", "val", "x", "y", "cur", "current"}:
        return "element"

    if n in {"arr", "array", "lst", "list", "items", "values", "nums", "numbers", "seq"}:
        return "sequence"

    if n in {"d", "dict", "map", "mapping", "table", "lookup"}:
        return "mapping"

    if n in {"flag", "ok", "valid", "found", "seen"}:
        return "boolean"

    if n in {"tmp", "temp"}:
        return "temporary"

    return "unknown"


def build_new_name_pool(
    old_name: str,
    role: str,
    used_identifiers: set[str],
    max_new_names: int,
) -> List[str]:
    role_pool = ROLE_POOLS.get(role, [])
    raw_pool = unique_keep_order(role_pool + GENERIC_POOL)

    out = []
    for new_name in raw_pool:
        if new_name == old_name:
            continue
        if new_name in used_identifiers:
            continue
        if not is_clean_identifier(new_name):
            continue
        out.append(new_name)

    return out[:max_new_names]


def get_local_binding_candidates(
    src: str,
    include_params: bool = False,
    min_occurrences: int = 1,
    max_new_names: int = 6,
) -> List[RenameCandidate]:
    """
    Returns one candidate per local binding group.

    Important:
    - This groups by scope + variable name, so same variable name in two functions
      becomes two different candidates.
    - first_start_char / first_end_char point to the first binding occurrence.
    """
    module = cst.parse_module(src)
    wrapper = MetadataWrapper(module)

    scope_map = wrapper.resolve(ScopeProvider)
    pos_map = wrapper.resolve(PositionProvider)

    used_identifiers = all_identifiers(src)

    # key = (scope_id, name)
    groups: Dict[Tuple[int, str], Dict[str, Any]] = {}
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

            node = assignment.node
            name_node: Optional[cst.Name] = None

            # Local variable / loop target / comprehension target.
            if isinstance(node, cst.Name):
                name_node = node

            # Optional function parameter.
            elif isinstance(node, cst.Param):
                if include_params and isinstance(node.name, cst.Name):
                    name_node = node.name
                else:
                    continue

            # Skip FunctionDef, ClassDef, tuple destructuring, etc.
            else:
                continue

            old_name = assignment.name
            if not is_clean_identifier(old_name):
                continue

            key = (id(scope), old_name)
            if key not in groups:
                groups[key] = {
                    "old_name": old_name,
                    "name_nodes": [],
                    "ref_nodes": [],
                }

            groups[key]["name_nodes"].append(name_node)

            for ref in assignment.references:
                if isinstance(ref.node, cst.Name):
                    groups[key]["ref_nodes"].append(ref.node)

    candidates: List[RenameCandidate] = []

    for (_scope_id, old_name), group in groups.items():
        all_nodes = group["name_nodes"] + group["ref_nodes"]
        unique_node_ids = {id(x) for x in all_nodes}
        occurrence_count = len(unique_node_ids)

        if occurrence_count < min_occurrences:
            continue

        # First binding occurrence, not first later reference.
        binding_nodes = group["name_nodes"]
        binding_nodes_sorted = sorted(
            binding_nodes,
            key=lambda n: (
                pos_map[n].start.line,
                pos_map[n].start.column,
            ),
        )
        first_node = binding_nodes_sorted[0]
        pos = pos_map[first_node]

        start = linecol_to_offset(src, pos.start.line, pos.start.column)
        end = linecol_to_offset(src, pos.end.line, pos.end.column)

        line_text = src.splitlines()[pos.start.line - 1].strip()
        role = infer_role(old_name, line_text)

        candidate_new_names = build_new_name_pool(
            old_name=old_name,
            role=role,
            used_identifiers=used_identifiers,
            max_new_names=max_new_names,
        )

        if not candidate_new_names:
            continue

        candidate_id = f"{old_name}@{start}:{end}"

        candidates.append(
            RenameCandidate(
                candidate_id=candidate_id,
                old_name=old_name,
                role=role,
                occurrence_count=occurrence_count,
                first_start_char=start,
                first_end_char=end,
                first_line=pos.start.line,
                first_column=pos.start.column,
                binding_context=line_text,
                candidate_new_names=candidate_new_names,
                original_continuation=old_name,
            )
        )

    # Deterministic order: earlier first appearance first.
    candidates.sort(key=lambda c: (c.first_start_char, c.old_name))
    return candidates


def make_source(example: Dict[str, Any], prefix_field: Optional[str], code_field: str) -> str:
    body = example.get(code_field, "")
    if body is None:
        body = ""

    if prefix_field:
        prefix = example.get(prefix_field, "")
        if prefix is None:
            prefix = ""
        return str(prefix) + str(body)

    return str(body)


def get_example_id(example: Dict[str, Any], i: int, id_field: Optional[str]) -> str:
    if id_field and id_field in example:
        return str(example[id_field])

    for key in ["task_id", "id", "problem_id", "name", "slug", "question_id"]:
        if key in example:
            return str(example[key])

    return str(i)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_examples(args) -> Iterable[Dict[str, Any]]:
    if args.input:
        return iter_jsonl(args.input)

    if args.hf_dataset:
        from datasets import load_dataset

        if args.hf_config:
            ds = load_dataset(args.hf_dataset, args.hf_config, split=args.split)
        else:
            ds = load_dataset(args.hf_dataset, split=args.split)
        return ds

    raise ValueError("Provide either --input or --hf_dataset.")


def main():
    parser = argparse.ArgumentParser()

    # Data source.
    parser.add_argument("--input", type=str, default=None, help="Local JSONL input.")
    parser.add_argument("--hf_dataset", type=str, default=None, help="HF dataset name.")
    parser.add_argument("--hf_config", type=str, default=None, help="Optional HF config.")
    parser.add_argument("--split", type=str, default="train", help="HF split name.")

    # Fields.
    parser.add_argument("--code_field", type=str, required=True)
    parser.add_argument("--prefix_field", type=str, default=None)
    parser.add_argument("--id_field", type=str, default=None)

    # Output.
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max_examples", type=int, default=None)

    # Candidate controls.
    parser.add_argument("--include_params", action="store_true")
    parser.add_argument("--min_occurrences", type=int, default=1)
    parser.add_argument("--max_new_names", type=int, default=6)

    # Usually keep this false for LLM curation to reduce prompt size.
    parser.add_argument("--include_prefix_for_scoring", action="store_true")

    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_parse_error = 0
    n_no_candidates = 0
    n_written = 0

    examples = load_examples(args)

    with open(out_path, "w", encoding="utf-8") as fout:
        for i, ex in enumerate(tqdm(examples, desc="generating candidates")):
            if args.max_examples is not None and i >= args.max_examples:
                break

            n_total += 1
            ex = dict(ex)

            example_id = get_example_id(ex, i, args.id_field)
            src = make_source(ex, args.prefix_field, args.code_field)

            if not src.strip():
                n_no_candidates += 1
                continue

            try:
                candidates = get_local_binding_candidates(
                    src=src,
                    include_params=args.include_params,
                    min_occurrences=args.min_occurrences,
                    max_new_names=args.max_new_names,
                )
            except Exception as e:
                n_parse_error += 1
                continue

            if not candidates:
                n_no_candidates += 1
                continue

            cand_dicts = []
            for c in candidates:
                d = asdict(c)

                # Prefix can be huge, but later scoring needs:
                # prefix = source[:first_start_char]
                if args.include_prefix_for_scoring:
                    d["prefix"] = src[: c.first_start_char]

                cand_dicts.append(d)

            row = {
                "example_id": example_id,
                "source_code": src,
                "num_candidates": len(cand_dicts),
                "candidates": cand_dicts,

                # Keep enough metadata to reconstruct later.
                "code_field": args.code_field,
                "prefix_field": args.prefix_field,
            }

            # Preserve useful benchmark fields if present.
            for key in ["entry_point", "test", "libs", "complete_prompt", "task_id"]:
                if key in ex:
                    row[key] = ex[key]

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    print("\nDone.")
    print(f"total examples:      {n_total}")
    print(f"written examples:    {n_written}")
    print(f"parse errors:        {n_parse_error}")
    print(f"no candidates:       {n_no_candidates}")
    print(f"output:              {out_path}")


if __name__ == "__main__":
    main()