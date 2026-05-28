import argparse
import gzip
import json
import random
import re
from collections import defaultdict

import pandas as pd


WORD_RE = re.compile(r"^[A-Za-z]+$")


def is_one_word(x):
    if not isinstance(x, str):
        return False
    x = x.strip()
    return bool(WORD_RE.fullmatch(x))


def load_swords(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def label_score(labels):
    """
    SWORDS substitute_labels are lists like TRUE / FALSE.
    TRUE_IMPLICIT is also counted as true if present.
    """
    if not labels:
        return 0.0
    true_count = sum(str(x).upper().startswith("TRUE") for x in labels)
    return true_count / len(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="swords-v1.1_test.json.gz")
    parser.add_argument("--output", required=True, help="output sampled csv")
    parser.add_argument("--all_output", default=None, help="optional output for all filtered rows")
    parser.add_argument("--n", type=int, default=1753)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_true", type=float, default=0.5)
    parser.add_argument(
        "--allow_repeated_target",
        action="store_true",
        help="Allow target word to occur multiple times in sentence",
    )
    args = parser.parse_args()

    swords = load_swords(args.input)

    # SWORDS format has contexts, targets, substitutes, substitute_labels.
    tid_to_sids = defaultdict(list)
    for sid, sub_obj in swords["substitutes"].items():
        tid_to_sids[sub_obj["target_id"]].append(sid)

    rows = []
    skipped = defaultdict(int)

    for tid, target_obj in swords["targets"].items():
        context_obj = swords["contexts"][target_obj["context_id"]]

        sentence = context_obj["context"]
        target = target_obj["target"].strip()
        offset = int(target_obj["offset"])
        pos = target_obj.get("pos", None)

        # Check offset correctness.
        if sentence[offset : offset + len(target)] != target:
            skipped["bad_offset"] += 1
            continue

        # Keep only one-word target.
        if not is_one_word(target):
            skipped["target_not_one_word"] += 1
            continue

        # Optional strictness: target appears only once.
        if not args.allow_repeated_target and sentence.count(target) != 1:
            skipped["target_repeated"] += 1
            continue

        prefix = sentence[:offset]
        suffix = sentence[offset + len(target) :]

        for sid in tid_to_sids[tid]:
            sub_obj = swords["substitutes"][sid]
            substitute = sub_obj["substitute"].strip()

            if not is_one_word(substitute):
                skipped["sub_not_one_word"] += 1
                continue

            if substitute.lower() == target.lower():
                skipped["same_as_target"] += 1
                continue

            labels = swords["substitute_labels"].get(sid, [])
            score = label_score(labels)

            if score < args.min_true:
                skipped["low_label_score"] += 1
                continue

            rows.append({
                "dataset": "swords",
                "source_target_id": tid,
                "source_substitute_id": sid,
                "clean_sentence": sentence,
                "target_clean": target,
                "substitute": substitute,
                "prefix": prefix,
                "suffix": suffix,
                "target_offset": offset,
                "pos": pos,
                "label_score": score,
                "num_labels": len(labels),
            })

    df = pd.DataFrame(rows)

    if args.all_output:
        df.to_csv(args.all_output, index=False)

    if len(df) < args.n:
        raise ValueError(
            f"Only {len(df)} filtered SWORDS rows, fewer than requested n={args.n}. "
            f"Try lowering --min_true or using --allow_repeated_target."
        )

    sampled = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    sampled.to_csv(args.output, index=False)

    print(f"All filtered SWORDS rows: {len(df)}")
    print(f"Sampled rows: {len(sampled)}")
    print(f"Saved sampled CSV to: {args.output}")

    if args.all_output:
        print(f"Saved all filtered CSV to: {args.all_output}")

    print("\nSkipped counts:")
    for k, v in sorted(skipped.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()