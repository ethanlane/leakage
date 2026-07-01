#!/usr/bin/env python3
"""
Build one-word synonym candidate pools for HumanEval and LiveCodeBench.

Usage:
  pip install datasets nltk tqdm

  python make_code_synonym_candidate_pools.py \
    --out_dir generated \
    --lcb_dataset jwu323/LiveCodeBench-v6-R182 \
    --lcb_min_date 2024-10-01

Outputs:
  generated/humaneval_synonym_candidates.jsonl
  generated/livecodebench_v6_r182_synonym_candidates.jsonl
  generated/*.skipped.jsonl
"""

import argparse
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm

try:
    import nltk
    from nltk.corpus import wordnet as wn
except Exception as e:
    raise RuntimeError(
        "Missing nltk. Install with: pip install nltk"
    ) from e


STOPWORDS = {
    # generic English
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while",
    "for", "to", "of", "in", "on", "at", "by", "from", "with", "without",
    "as", "is", "are", "was", "were", "be", "been", "being", "can", "could",
    "should", "would", "will", "may", "might", "must", "do", "does", "did",
    "this", "that", "these", "those", "it", "its", "you", "your", "we", "our",
    "they", "their", "there", "here",

    # coding benchmark boilerplate
    "given", "write", "function", "program", "return", "returns", "input",
    "output", "example", "examples", "note", "constraints", "constraint",
    "test", "case", "cases", "class", "method", "implement", "solution",

    # too central / dangerous in coding tasks
    "integer", "integers", "number", "numbers", "string", "strings",
    "array", "arrays", "list", "lists", "matrix", "tree", "graph",
    "node", "nodes", "index", "indices", "value", "values",
}


# Words that WordNet often gives bad code-problem substitutes for.
BAD_TARGETS = {
    "true", "false", "none", "null",
    "python", "leetcode", "codeforces", "atcoder",
    "modulo", "mod", "xor",
}


QUESTION_FIELD_CANDIDATES = [
    "question_content",
    "question",
    "problem_statement",
    "statement",
    "description",
    "prompt",
    "content",
]

STARTER_FIELD_CANDIDATES = [
    "starter_code",
    "starter",
    "code",
    "declaration",
    "function_signature",
]

ID_FIELD_CANDIDATES = [
    "question_id",
    "task_id",
    "id",
    "problem_id",
    "contest_id",
    "question_title",
    "title",
    "name",
]

DATE_FIELD_CANDIDATES = [
    "release_date",
    "contest_date",
    "date",
    "published_date",
    "publication_date",
]


def ensure_wordnet() -> None:
    try:
        _ = wn.synsets("test")
    except LookupError:
        nltk.download("wordnet")
        nltk.download("omw-1.4")


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_string_field(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False)


def get_first_existing(record: Dict[str, Any], candidates: List[str]) -> Tuple[Optional[str], str]:
    for k in candidates:
        if k in record and record[k] not in (None, ""):
            return k, normalize_string_field(record[k])
    return None, ""


def parse_date_maybe(x: Any) -> Optional[datetime]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None

    # Handles "2024-10-01", "2024-10-01T00:00:00", etc.
    s = s.replace("Z", "")
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s[:19] if "T" in s else s[:10], fmt)
        except ValueError:
            pass

    m = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
    if m:
        y, mo, d = map(int, m.groups())
        return datetime(y, mo, d)

    return None


def extract_humaneval_docstring(prompt: str) -> str:
    """
    HumanEval prompt usually contains a function signature plus a triple-quoted docstring.
    We choose target words from the docstring, but offsets are later mapped back to full prompt.
    """
    m = re.search(r'("""|\'\'\')([\s\S]*?)(\1)', prompt)
    if m:
        return m.group(2)
    return prompt


def one_word_synonyms(word: str, max_candidates: int = 20) -> List[str]:
    w = word.lower()
    out = []

    for syn in wn.synsets(w):
        for lemma in syn.lemmas():
            cand = lemma.name().replace("_", " ").strip().lower()

            # We want one-token lexical substitutions only.
            if " " in cand or "-" in cand:
                continue
            if not re.fullmatch(r"[a-z]+", cand):
                continue
            if cand == w:
                continue
            if len(cand) < 3:
                continue
            if cand in STOPWORDS or cand in BAD_TARGETS:
                continue
            if cand not in out:
                out.append(cand)

    return out[:max_candidates]


def token_spans(text: str) -> List[Tuple[str, int, int]]:
    # English words only; avoid code identifiers with underscores/numbers.
    return [
        (m.group(0), m.start(), m.end())
        for m in re.finditer(r"\b[A-Za-z]{3,}\b", text)
    ]


def choose_target_word(question_text: str, min_candidates: int = 3) -> Optional[Dict[str, Any]]:
    """
    Deterministically choose one target word with a non-trivial synonym pool.
    Preference: content words with more synonym candidates and longer length.
    """
    scored = []

    for tok, start, end in token_spans(question_text):
        lower = tok.lower()

        if lower in STOPWORDS or lower in BAD_TARGETS:
            continue
        if len(lower) < 4:
            continue

        syns = one_word_synonyms(lower)
        if len(syns) < min_candidates:
            continue

        # Prefer less boilerplate-ish content words.
        score = 10 * len(syns) + min(len(lower), 12)

        # Slightly prefer adjectives/adverbs/verbs often found in problem statements.
        if lower.endswith(("able", "ible", "ive", "al", "ous", "ful", "less", "ing", "ed")):
            score += 5

        scored.append({
            "target": tok,
            "target_lower": lower,
            "start": start,
            "end": end,
            "synonyms": syns,
            "score": score,
        })

    if not scored:
        return None

    scored.sort(key=lambda x: (-x["score"], x["start"]))
    best = scored[0]
    best.pop("score", None)
    return best


def map_question_offset_to_full_prompt(
    full_prompt: str,
    question_text: str,
    q_start: int,
    q_end: int,
) -> Tuple[int, int]:
    """
    Most cases: question_text is literally inside full_prompt.
    If not, assume question_text begins at offset 0.
    """
    base = full_prompt.find(question_text)
    if base < 0:
        base = 0
    return base + q_start, base + q_end


def build_humaneval_rows(min_candidates: int, max_rows: Optional[int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ds = load_dataset("openai/openai_humaneval", split="test")
    rows, skipped = [], []

    for rec in tqdm(ds, desc="HumanEval"):
        rec = dict(rec)
        task_id = rec.get("task_id")
        full_prompt = rec["prompt"]
        question_text = extract_humaneval_docstring(full_prompt)

        chosen = choose_target_word(question_text, min_candidates=min_candidates)
        if chosen is None:
            skipped.append({
                "dataset": "humaneval",
                "task_id": task_id,
                "reason": "no_target_with_enough_synonyms",
                "question_text": question_text,
            })
            continue

        full_start, full_end = map_question_offset_to_full_prompt(
            full_prompt,
            question_text,
            chosen["start"],
            chosen["end"],
        )

        rows.append({
            "dataset": "humaneval",
            "source_dataset": "openai/openai_humaneval",
            "task_id": task_id,
            "entry_point": rec.get("entry_point"),
            "full_prompt": full_prompt,
            "question_text": question_text,

            "target": full_prompt[full_start:full_end],
            "target_lower": chosen["target_lower"],
            "target_offset": full_start,
            "target_end": full_end,
            "target_offset_in_question": chosen["start"],
            "target_end_in_question": chosen["end"],

            "prefix": full_prompt[:full_start],
            "suffix": full_prompt[full_end:],
            "candidate_substitutes": chosen["synonyms"],
            "num_candidates": len(chosen["synonyms"]),

            # Keep lightweight metadata only.
            "metadata": {
                "canonical_solution": rec.get("canonical_solution"),
                "test": rec.get("test"),
            },
        })

        if max_rows is not None and len(rows) >= max_rows:
            break

    return rows, skipped


def build_lcb_rows(
    lcb_dataset: str,
    lcb_split: Optional[str],
    lcb_min_date: Optional[str],
    min_candidates: int,
    max_rows: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if lcb_split:
        ds = load_dataset(lcb_dataset, split=lcb_split)
    else:
        loaded = load_dataset(lcb_dataset)
        # Prefer test if available; otherwise first split.
        split_name = "test" if "test" in loaded else list(loaded.keys())[0]
        ds = loaded[split_name]

    min_dt = parse_date_maybe(lcb_min_date) if lcb_min_date else None

    rows, skipped = [], []

    for idx, rec in enumerate(tqdm(ds, desc="LiveCodeBench")):
        rec = dict(rec)

        id_key, source_id = get_first_existing(rec, ID_FIELD_CANDIDATES)
        if not source_id:
            source_id = str(idx)

        date_key, date_s = get_first_existing(rec, DATE_FIELD_CANDIDATES)
        rec_dt = parse_date_maybe(date_s)

        if min_dt is not None:
            if rec_dt is None:
                skipped.append({
                    "dataset": "livecodebench",
                    "source_id": source_id,
                    "reason": "missing_or_unparseable_date_for_filter",
                    "date_field": date_key,
                    "date_value": date_s,
                })
                continue
            if rec_dt < min_dt:
                continue

        q_key, question_text = get_first_existing(rec, QUESTION_FIELD_CANDIDATES)
        if not question_text:
            skipped.append({
                "dataset": "livecodebench",
                "source_id": source_id,
                "reason": "no_question_field_found",
                "available_fields": list(rec.keys()),
            })
            continue

        starter_key, starter_code = get_first_existing(rec, STARTER_FIELD_CANDIDATES)

        # Full prompt used for substitution/scoring.
        # The target is chosen only inside question_text, but suffix can include starter code too.
        full_prompt = question_text
        if starter_code:
            full_prompt = question_text.rstrip() + "\n\n" + starter_code.lstrip()

        chosen = choose_target_word(question_text, min_candidates=min_candidates)
        if chosen is None:
            skipped.append({
                "dataset": "livecodebench",
                "source_id": source_id,
                "reason": "no_target_with_enough_synonyms",
                "question_field": q_key,
                "question_text": question_text[:2000],
            })
            continue

        full_start, full_end = map_question_offset_to_full_prompt(
            full_prompt,
            question_text,
            chosen["start"],
            chosen["end"],
        )

        rows.append({
            "dataset": "livecodebench",
            "source_dataset": lcb_dataset,
            "source_id": source_id,
            "id_field": id_key,
            "date_field": date_key,
            "date": date_s,
            "question_field": q_key,
            "starter_code_field": starter_key,

            "full_prompt": full_prompt,
            "question_text": question_text,

            "target": full_prompt[full_start:full_end],
            "target_lower": chosen["target_lower"],
            "target_offset": full_start,
            "target_end": full_end,
            "target_offset_in_question": chosen["start"],
            "target_end_in_question": chosen["end"],

            "prefix": full_prompt[:full_start],
            "suffix": full_prompt[full_end:],
            "candidate_substitutes": chosen["synonyms"],
            "num_candidates": len(chosen["synonyms"]),

            "metadata": {
                # Keep common LCB fields if present.
                "platform": rec.get("platform"),
                "question_title": rec.get("question_title") or rec.get("title"),
                "difficulty": rec.get("difficulty"),
                "contest_id": rec.get("contest_id"),
            },
        })

        if max_rows is not None and len(rows) >= max_rows:
            break

    return rows, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="generated")
    parser.add_argument("--lcb_dataset", default="jwu323/LiveCodeBench-v6-R182")
    parser.add_argument("--lcb_split", default=None)
    parser.add_argument(
        "--lcb_min_date",
        default="2024-10-01",
        help="For Qwen2.5-Coder-7B-Base, 2024-10-01 is a conservative post-release unseen cutoff.",
    )
    parser.add_argument("--min_candidates", type=int, default=3)
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    ensure_wordnet()

    he_rows, he_skipped = build_humaneval_rows(
        min_candidates=args.min_candidates,
        max_rows=args.max_rows,
    )

    lcb_rows, lcb_skipped = build_lcb_rows(
        lcb_dataset=args.lcb_dataset,
        lcb_split=args.lcb_split,
        lcb_min_date=args.lcb_min_date,
        min_candidates=args.min_candidates,
        max_rows=args.max_rows,
    )

    he_out = os.path.join(args.out_dir, "humaneval_synonym_candidates.jsonl")
    he_skip_out = os.path.join(args.out_dir, "humaneval_synonym_candidates.skipped.jsonl")
    lcb_out = os.path.join(args.out_dir, "livecodebench_v6_r182_synonym_candidates.jsonl")
    lcb_skip_out = os.path.join(args.out_dir, "livecodebench_v6_r182_synonym_candidates.skipped.jsonl")

    write_jsonl(he_out, he_rows)
    write_jsonl(he_skip_out, he_skipped)
    write_jsonl(lcb_out, lcb_rows)
    write_jsonl(lcb_skip_out, lcb_skipped)

    print("Done.")
    print(f"HumanEval accepted: {len(he_rows)}")
    print(f"HumanEval skipped:  {len(he_skipped)}")
    print(f"LCB accepted:       {len(lcb_rows)}")
    print(f"LCB skipped:        {len(lcb_skipped)}")
    print()
    print(f"Wrote: {he_out}")
    print(f"Wrote: {he_skip_out}")
    print(f"Wrote: {lcb_out}")
    print(f"Wrote: {lcb_skip_out}")


if __name__ == "__main__":
    main()