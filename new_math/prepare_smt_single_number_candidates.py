#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?![\w.])")


DANGER_KEYWORDS = [
    # conventions / fixed definitions
    "digit", "digits", "base", "mod", "modulo", "remainder",
    "standard die", "standard dice",

    # indices / labels / references
    "figure", "diagram", "problem", "case", "part", "chapter",

    # frequently special-number dependent
    "prime", "divisible", "factor", "multiple", "gcd", "lcm",
    "integer", "positive integer", "nonnegative integer",

    # geometry can be valid, but many contest geometry numbers are constrained
    "triangle inequality", "regular polygon", "sides", "vertices",
]


LABEL_LEFT_PATTERNS = [
    r"x_\s*$",
    r"y_\s*$",
    r"a_\s*$",
    r"b_\s*$",
    r"c_\s*$",
    r"n_\s*$",
    r"\\angle\s*$",
]


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def load_smt_2024() -> List[Dict[str, Any]]:
    ds = load_dataset("nmayorga7/smt-2024", split="train")
    rows = []

    for r in ds:
        rows.append({
            "dataset": "smt_2024",
            "source_id": str(r.get("id", "")),
            "problem": r["question"],
            "answer": str(r.get("answer", "")),
            "raw_category": str(r.get("category", "")),
            "raw_problem_type": [],
        })

    return rows


def load_smt_2025() -> List[Dict[str, Any]]:
    ds = load_dataset("MathArena/smt_2025", split="train")
    rows = []

    for r in ds:
        problem_type = r.get("problem_type", [])
        if problem_type is None:
            problem_type = []
        if isinstance(problem_type, str):
            problem_type = [problem_type]

        rows.append({
            "dataset": "smt_2025",
            "source_id": str(r.get("problem_idx", "")),
            "problem": r["problem"],
            "answer": str(r.get("answer", "")),
            "raw_category": "",
            "raw_problem_type": list(problem_type),
        })

    return rows


def normalize_category(row: Dict[str, Any]) -> str:
    """
    Rough mapping into Hendrycks-MATH-style categories.
    Final category should be checked after semantic filtering.
    """

    text = normalize_space(row["problem"]).lower()
    labels = " ".join([row.get("raw_category", "")] + row.get("raw_problem_type", [])).lower()

    combined = labels + " " + text

    if "geometry" in combined or any(w in combined for w in ["circle", "triangle", "square", "polygon", "radius", "area", "perimeter"]):
        return "geometry"

    if "calculus" in combined or any(w in combined for w in ["derivative", "integral", "limit"]):
        return "reject_calculus_no_clean_hendricks_match"

    if any(w in combined for w in ["probability", "expected", "choose", "ways", "permutation", "combination", "arrangements"]):
        return "counting_and_probability"

    if any(w in combined for w in ["prime", "divisible", "modulo", "remainder", "integer", "gcd", "lcm"]):
        return "number_theory"

    if any(w in combined for w in ["polynomial", "quadratic", "equation", "function", "roots", "system"]):
        return "algebra"

    if "algebra" in combined:
        return "algebra"

    if "discrete" in combined:
        return "counting_and_probability"

    # General/Guts are contest sections, not real subject labels.
    return "unknown_general_or_guts"


def find_numbers(text: str) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in NUMBER_RE.finditer(text)]


def perturb_number(num: str) -> Optional[str]:
    raw = num.replace(",", "")

    try:
        if "." in raw:
            x = float(raw)
            if not math.isfinite(x):
                return None
            decimals = len(raw.split(".")[-1])
            y = x + 1.0
            return f"{y:.{decimals}f}"

        x = int(raw)

        # Conservative: avoid super-special constants.
        if x in {0, 1}:
            return None

        # Avoid years.
        if 1900 <= abs(x) <= 2099:
            return None

        if x > 0:
            return str(x + 1)
        else:
            return str(x - 1)

    except Exception:
        return None


def replace_span(text: str, start: int, end: int, repl: str) -> str:
    return text[:start] + repl + text[end:]


def exact_number_occurrence_count(text: str, num: str) -> int:
    pat = re.compile(rf"(?<![\w.]){re.escape(num)}(?![\w.])")
    return len(pat.findall(text))


def window(text: str, start: int, end: int, k: int = 45) -> str:
    return text[max(0, start - k): min(len(text), end + k)]


def auto_reject_reason(text: str, start: int, end: int, num: str, pert: str) -> Optional[str]:
    left = text[max(0, start - 20):start]
    right = text[end:min(len(text), end + 20)]
    local = window(text, start, end, 60).lower()

    # Subscripts, superscripts, labels, LaTeX powers.
    if start > 0 and text[start - 1] in "_^#":
        return "index_or_exponent_like_position"

    if end < len(text) and text[end:end + 2] in ["st", "nd", "rd", "th"]:
        return "ordinal_number"

    if "/" in left[-3:] or "/" in right[:3]:
        return "fraction_or_ratio_nearby"

    if "\\frac" in left[-10:] or "\\frac" in local:
        return "latex_fraction_nearby"

    if re.search(r"[A-Za-z]$", left) or re.search(r"^[A-Za-z]", right):
        return "attached_to_word_or_variable"

    for pat in LABEL_LEFT_PATTERNS:
        if re.search(pat, left):
            return "label_or_subscript_like_number"

    if exact_number_occurrence_count(text, num) != 1:
        return "same_number_string_appears_multiple_times"

    raw_int = None
    try:
        raw_int = int(num.replace(",", ""))
    except Exception:
        pass

    if raw_int is not None:
        if raw_int in {0, 1}:
            return "special_constant_0_or_1"

        if 1900 <= abs(raw_int) <= 2099:
            return "year_like_number"

    for kw in DANGER_KEYWORDS:
        if kw in local:
            return f"danger_keyword_nearby:{kw}"

    return None


def single_token_info(tokenizer, text: str, start: int, end: int, max_length: int) -> Optional[Dict[str, Any]]:
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )

    offsets = enc["offset_mapping"]
    input_ids = enc["input_ids"]

    overlapping = []
    for i, (a, b) in enumerate(offsets):
        if a == b == 0:
            continue
        if b > start and a < end:
            overlapping.append(i)

    if len(overlapping) != 1:
        return None

    tok_idx = overlapping[0]
    tok_id = input_ids[tok_idx]
    decoded = tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)

    return {
        "token_index": tok_idx,
        "token_id": tok_id,
        "decoded_token": decoded,
    }


def make_candidates_for_row(
    row: Dict[str, Any],
    tokenizer,
    max_length: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

    accepted = []
    rejected = []

    text = row["problem"]
    norm_cat = normalize_category(row)

    for start, end, num in find_numbers(text):
        pert = perturb_number(num)

        base = {
            "dataset": row["dataset"],
            "source_id": row["source_id"],
            "answer": row.get("answer", ""),
            "raw_category": row.get("raw_category", ""),
            "raw_problem_type": row.get("raw_problem_type", []),
            "norm_category": norm_cat,
            "original_problem": text,
            "changed_start": start,
            "changed_end": end,
            "original_number": num,
            "perturbed_number": pert,
            "local_context": window(text, start, end, 80),
        }

        if pert is None:
            base["reject_reason"] = "no_valid_numeric_perturbation"
            rejected.append(base)
            continue

        reason = auto_reject_reason(text, start, end, num, pert)
        if reason is not None:
            base["reject_reason"] = reason
            rejected.append(base)
            continue

        pert_text = replace_span(text, start, end, pert)
        pert_end = start + len(pert)

        orig_tok = single_token_info(tokenizer, text, start, end, max_length)
        if orig_tok is None:
            base["reject_reason"] = "original_number_not_single_token"
            rejected.append(base)
            continue

        pert_tok = single_token_info(tokenizer, pert_text, start, pert_end, max_length)
        if pert_tok is None:
            base["reject_reason"] = "perturbed_number_not_single_token"
            rejected.append(base)
            continue

        cand = dict(base)
        cand.update({
            "perturbed_problem": pert_text,
            "perturbed_changed_start": start,
            "perturbed_changed_end": pert_end,
            "orig_token_id": orig_tok["token_id"],
            "orig_decoded_token": orig_tok["decoded_token"],
            "pert_token_id": pert_tok["token_id"],
            "pert_decoded_token": pert_tok["decoded_token"],
            "auto_filter": "ACCEPT_FOR_SEMANTIC_REVIEW",
            "judge_label": "",
            "judge_reason": "",
        })
        accepted.append(cand)

    return accepted, rejected


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return

    fieldnames = [
        "dataset",
        "source_id",
        "norm_category",
        "raw_category",
        "raw_problem_type",
        "original_number",
        "perturbed_number",
        "changed_start",
        "changed_end",
        "orig_decoded_token",
        "pert_decoded_token",
        "local_context",
        "original_problem",
        "perturbed_problem",
        "judge_label",
        "judge_reason",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="agentica-org/DeepScaleR-1.5B-Preview")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--out-prefix", default="result_num/smt_2024_2025_single_number_auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    rows = []
    rows.extend(load_smt_2024())
    rows.extend(load_smt_2025())

    all_candidates = []
    all_rejects = []

    for row in tqdm(rows, desc="building SMT candidates"):
        cands, rejects = make_candidates_for_row(
            row=row,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
        all_candidates.extend(cands)
        all_rejects.extend(rejects)

    random.shuffle(all_candidates)

    jsonl_path = args.out_prefix + ".candidates.jsonl"
    csv_path = args.out_prefix + ".candidates.csv"
    reject_path = args.out_prefix + ".rejects.jsonl"
    summary_path = args.out_prefix + ".summary.json"

    write_jsonl(jsonl_path, all_candidates)
    write_csv(csv_path, all_candidates)
    write_jsonl(reject_path, all_rejects)

    summary = {
        "n_problems_total": len(rows),
        "n_candidates_auto_accept": len(all_candidates),
        "n_rejected_numeric_spans": len(all_rejects),
        "accepted_by_dataset": dict(Counter(r["dataset"] for r in all_candidates)),
        "accepted_by_norm_category": dict(Counter(r["norm_category"] for r in all_candidates)),
        "rejected_by_reason": dict(Counter(r.get("reject_reason", "unknown") for r in all_rejects)),
        "outputs": {
            "candidates_jsonl": jsonl_path,
            "candidates_csv": csv_path,
            "rejects_jsonl": reject_path,
            "summary_json": summary_path,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()