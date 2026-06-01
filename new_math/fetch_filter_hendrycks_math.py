#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict

import pandas as pd
from datasets import load_dataset, get_dataset_config_names


DATASET_NAME = "EleutherAI/hendrycks_math"

BAD_FIGURE_PATTERNS = [
    r"\[asy\]",
    r"begin\{asy\}",
    r"end\{asy\}",
    r"unitsize",
    r"\bdraw\(",
    r"includegraphics",
    r"diagram",
    r"figure",
]

# Conservative numeric matcher: integers/decimals, optional sign, optional commas.
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?(?![A-Za-z])")


def normalize_type(x):
    return str(x).strip().lower().replace(" ", "_")


def has_bad_figure_reference(text):
    low = text.lower()
    return any(re.search(pat, low) for pat in BAD_FIGURE_PATTERNS)


def clean_number_string(s):
    return s.replace(",", "")


def is_finite_number(s):
    try:
        float(clean_number_string(s))
        return True
    except Exception:
        return False


def occurrence_count(text, number_str):
    return len(re.findall(re.escape(number_str), text))


def is_probably_metadata_or_formatting(text, start, end):
    """
    Heuristic only. We keep many borderline candidates for later LLM review,
    but remove obvious formatting/code artifacts.
    """
    left = text[max(0, start - 20):start]
    right = text[end:min(len(text), end + 20)]

    # LaTeX/Asymptote sizing and drawing constants are risky.
    local = (left + text[start:end] + right).lower()
    bad_local = ["unitsize", "size(", "draw(", "label(", "dot(", "fontsize"]
    if any(x in local for x in bad_local):
        return True

    # Section/problem labels are not math quantities.
    if re.search(r"(problem|example|figure|diagram)\s*$", left.lower()):
        return True

    return False


def propose_natural_perturbations(num_str):
    """
    Generate natural same-class numeric alternatives.
    These are only proposals for later LLM/math validation, not final decisions.
    """
    raw = clean_number_string(num_str)

    try:
        if "." in raw:
            x = float(raw)
            is_int = False
        else:
            x = int(raw)
            is_int = True
    except Exception:
        return []

    proposals = []

    if not is_int:
        # Simple decimal perturbations.
        for mult in [1.2, 0.8, 1.5]:
            y = round(x * mult, 2)
            if y != x and y > 0:
                proposals.append(str(y))
        return dedupe_keep_order(proposals)

    sign = -1 if x < 0 else 1
    a = abs(x)

    if a == 0:
        proposals = ["1", "2", "-1"]
    elif a <= 10:
        # Small integers: nearby values are usually natural.
        candidates = [a + 1, max(1, a - 1), a + 2]
        proposals = [str(sign * c) for c in candidates if c != a]
    elif a <= 30:
        # Contest-natural small/medium values.
        candidates = [a + 2, a - 2, a + 5, max(1, a - 5)]
        proposals = [str(sign * c) for c in candidates if c > 0 and c != a]
    elif a <= 100:
        # Prefer multiples / clean nearby numbers.
        candidates = []
        if a % 5 == 0:
            candidates += [a + 5, a - 5, a + 10, max(5, a - 10)]
        else:
            candidates += [round_to_base(a, 5), round_to_base(a, 10), a + 1]
        proposals = [str(sign * c) for c in candidates if c > 0 and c != a]
    else:
        # Round large numbers should stay round.
        candidates = []
        if a % 100 == 0:
            candidates += [int(a * 1.2), int(a * 0.8), a + 100, max(100, a - 100)]
        elif a % 10 == 0:
            candidates += [a + 10, max(10, a - 10), int(round(a * 1.2 / 10) * 10)]
        else:
            candidates += [a + 1, a - 1]
        proposals = [str(sign * c) for c in candidates if c > 0 and c != a]

    return dedupe_keep_order(proposals)[:5]


def round_to_base(x, base):
    return int(base * round(float(x) / base))


def dedupe_keep_order(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_number_candidates(problem, max_candidates=20):
    candidates = []

    for m in NUMBER_RE.finditer(problem):
        s = m.group(0)

        if not is_finite_number(s):
            continue

        if is_probably_metadata_or_formatting(problem, m.start(), m.end()):
            continue

        proposals = propose_natural_perturbations(s)
        if not proposals:
            continue

        candidates.append({
            "number": s,
            "start": m.start(),
            "end": m.end(),
            "left_context": problem[max(0, m.start() - 80):m.start()],
            "right_context": problem[m.end():min(len(problem), m.end() + 80)],
            "occurrence_count": occurrence_count(problem, s),
            "proposal_numbers": proposals,
        })

        if len(candidates) >= max_candidates:
            break

    return candidates


def largest_remainder_quotas(counts, total):
    """
    Allocate total samples proportionally to counts using largest remainder.
    """
    keys = sorted(counts)
    s = sum(counts.values())
    raw = {k: total * counts[k] / s for k in keys}
    quotas = {k: int(math.floor(raw[k])) for k in keys}

    remaining = total - sum(quotas.values())
    frac_order = sorted(keys, key=lambda k: raw[k] - quotas[k], reverse=True)

    for k in frac_order[:remaining]:
        quotas[k] += 1

    return quotas


def cap_and_redistribute_quotas(quotas, available, total):
    """
    If a type lacks enough filtered rows, cap it and redistribute.
    """
    quotas = dict(quotas)
    keys = sorted(quotas)

    changed = True
    while changed:
        changed = False

        deficit = 0
        for k in keys:
            if quotas[k] > available.get(k, 0):
                deficit += quotas[k] - available.get(k, 0)
                quotas[k] = available.get(k, 0)
                changed = True

        if deficit == 0:
            break

        room_keys = [k for k in keys if quotas[k] < available.get(k, 0)]
        if not room_keys:
            break

        # Redistribute one by one to types with most remaining capacity.
        for _ in range(deficit):
            room_keys = sorted(
                [k for k in keys if quotas[k] < available.get(k, 0)],
                key=lambda k: available.get(k, 0) - quotas[k],
                reverse=True,
            )
            if not room_keys:
                break
            quotas[room_keys[0]] += 1

    # If still not total because not enough available, caller will sample fewer.
    return quotas


def fetch_rows(args):
    configs = args.configs
    if configs is None:
        configs = get_dataset_config_names(DATASET_NAME)

    rows = []

    for config in configs:
        ds = load_dataset(DATASET_NAME, config)

        for split in args.splits:
            if split not in ds:
                continue

            for i, r in enumerate(ds[split]):
                problem = str(r.get("problem", ""))
                solution = str(r.get("solution", ""))
                level = str(r.get("level", ""))
                typ = normalize_type(r.get("type", config))

                rows.append({
                    "dataset": "hendrycks_math",
                    "source_id": f"{config}/{split}/{i}",
                    "config": config,
                    "split": split,
                    "normalized_category": typ,
                    "level": level,
                    "problem": problem,
                    "solution": solution,
                })

    return rows


def filter_rows(rows, args):
    filtered = []
    skip = Counter()

    for r in rows:
        problem = r["problem"]

        if len(problem) < args.min_chars:
            skip["too_short"] += 1
            continue

        if len(problem) > args.max_chars:
            skip["too_long"] += 1
            continue

        if args.exclude_figures and has_bad_figure_reference(problem):
            skip["figure_or_diagram"] += 1
            continue

        candidates = extract_number_candidates(
            problem,
            max_candidates=args.max_candidates_per_problem,
        )

        if len(candidates) < args.min_candidates_per_problem:
            skip["too_few_numeric_candidates"] += 1
            continue

        rr = dict(r)
        rr["candidate_numbers_json"] = json.dumps(candidates, ensure_ascii=False)
        rr["num_numeric_candidates"] = len(candidates)
        filtered.append(rr)

    return filtered, skip


def stratified_sample(df, total, seed, reference_counts=None):
    if reference_counts is None:
        reference_counts = Counter(df["normalized_category"])

    available = Counter(df["normalized_category"])
    quotas = largest_remainder_quotas(reference_counts, total)
    quotas = cap_and_redistribute_quotas(quotas, available, total)

    sampled_parts = []

    for typ, n in quotas.items():
        if n <= 0:
            continue
        g = df[df["normalized_category"] == typ]
        if len(g) == 0:
            continue
        sampled_parts.append(g.sample(n=min(n, len(g)), random_state=seed))

    if sampled_parts:
        out = pd.concat(sampled_parts, ignore_index=True)
    else:
        out = pd.DataFrame(columns=df.columns)

    # If rounding/capping left us short, fill from remaining rows.
    if len(out) < total:
        used = set(out["source_id"])
        remaining = df[~df["source_id"].isin(used)]
        need = total - len(out)
        if len(remaining) > 0:
            extra = remaining.sample(n=min(need, len(remaining)), random_state=seed + 1)
            out = pd.concat([out, extra], ignore_index=True)

    return out.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True), quotas


def write_jsonl(df, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            obj = row.to_dict()
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--sample_out", default="hendrycks_math_filtered_70.jsonl")
    p.add_argument("--filtered_out", default="hendrycks_math_filtered_all.jsonl")
    p.add_argument("--summary_out", default="hendrycks_math_filter_summary.json")

    p.add_argument("--n", type=int, default=70)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        help="Use test by default. Use --splits train test for all.",
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Optional subset configs. Default: all configs.",
    )

    p.add_argument("--min_chars", type=int, default=40)
    p.add_argument("--max_chars", type=int, default=1600)
    p.add_argument("--exclude_figures", action="store_true", default=True)
    p.add_argument("--keep_figures", dest="exclude_figures", action="store_false")

    p.add_argument("--min_candidates_per_problem", type=int, default=1)
    p.add_argument("--max_candidates_per_problem", type=int, default=20)

    args = p.parse_args()

    rows = fetch_rows(args)
    all_df = pd.DataFrame(rows)

    filtered, skip = filter_rows(rows, args)
    filtered_df = pd.DataFrame(filtered)

    if len(filtered_df) == 0:
        raise ValueError("No rows left after filtering. Relax filters.")

    # Use the fetched dataset distribution as the reference distribution.
    reference_counts = Counter(all_df["normalized_category"])
    sampled_df, quotas = stratified_sample(
        filtered_df,
        total=args.n,
        seed=args.seed,
        reference_counts=reference_counts,
    )

    write_jsonl(filtered_df, args.filtered_out)
    write_jsonl(sampled_df, args.sample_out)

    summary = {
        "dataset": DATASET_NAME,
        "splits": args.splits,
        "n_raw": int(len(all_df)),
        "n_filtered": int(len(filtered_df)),
        "n_sampled": int(len(sampled_df)),
        "raw_type_counts": dict(Counter(all_df["normalized_category"])),
        "filtered_type_counts": dict(Counter(filtered_df["normalized_category"])),
        "sampled_type_counts": dict(Counter(sampled_df["normalized_category"])),
        "target_quotas": dict(quotas),
        "skip_counts": dict(skip),
        "filters": {
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "exclude_figures": args.exclude_figures,
            "min_candidates_per_problem": args.min_candidates_per_problem,
            "max_candidates_per_problem": args.max_candidates_per_problem,
        },
    }

    os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved filtered all: {args.filtered_out}")
    print(f"Saved sampled 70:   {args.sample_out}")
    print(f"Saved summary:      {args.summary_out}")


if __name__ == "__main__":
    main()