import argparse
import ast
import re
import pandas as pd


WORD_RE = re.compile(r"^[A-Za-z]+$")


def is_one_word(x):
    if not isinstance(x, str):
        return False
    x = x.strip()
    return bool(WORD_RE.fullmatch(x))


def parse_subs(x):
    """
    Parse ProLex substitute columns like acc_subs / prof_acc_subs.
    Handles Python-list strings and simple comma-separated fallback.
    """
    if pd.isna(x):
        return []

    s = str(x).strip()

    try:
        items = ast.literal_eval(s)
        if not isinstance(items, list):
            items = [items]
    except Exception:
        s = s.strip("[]")
        items = [p.strip() for p in s.split(",") if p.strip()]

    cleaned = []
    for item in items:
        if isinstance(item, (list, tuple)):
            item = item[0]

        item = str(item).strip().strip("'").strip('"')

        # remove trailing CEFR/proficiency notes like "word (B2)"
        item = re.sub(r"\s*\([^)]*\)\s*$", "", item).strip()

        if item:
            cleaned.append(item)

    return cleaned


def get_marked_target(sentence, fallback_target):
    """
    ProLex sentences usually mark target like **word**.
    Prefer the marked target if available.
    """
    sentence = str(sentence)
    m = re.search(r"\*\*(.+?)\*\*", sentence)
    if m:
        return m.group(1).strip()
    return str(fallback_target).strip()


def remove_target_markers(sentence):
    return str(sentence).replace("**", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Original ProLex csv")
    parser.add_argument("--output", required=True, help="Filtered output csv")
    parser.add_argument(
        "--sub_col",
        default="acc_subs",
        help="Which substitute column to use, e.g. acc_subs or prof_acc_subs",
    )
    parser.add_argument(
        "--explode",
        action="store_true",
        help="Save one row per valid substitute instead of one row per original example",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if "Sentence" not in df.columns:
        raise ValueError(f"Cannot find Sentence column. Columns: {list(df.columns)}")

    if "target word" in df.columns:
        target_col = "target word"
    elif "target_word" in df.columns:
        target_col = "target_word"
    elif "target" in df.columns:
        target_col = "target"
    else:
        raise ValueError(f"Cannot find target column. Columns: {list(df.columns)}")

    if args.sub_col not in df.columns:
        raise ValueError(f"Cannot find {args.sub_col}. Columns: {list(df.columns)}")

    rows = []

    for idx, row in df.iterrows():
        sentence = str(row["Sentence"])
        target = get_marked_target(sentence, row[target_col])

        # filter target word
        if not is_one_word(target):
            continue

        subs = parse_subs(row[args.sub_col])

        # filter substitute words
        one_word_subs = []
        for sub in subs:
            sub = sub.strip()
            if is_one_word(sub) and sub.lower() != target.lower():
                one_word_subs.append(sub)

        # remove duplicates while preserving order
        seen = set()
        one_word_subs = [
            s for s in one_word_subs
            if not (s.lower() in seen or seen.add(s.lower()))
        ]

        if len(one_word_subs) == 0:
            continue

        clean_sentence = remove_target_markers(sentence)

        if args.explode:
            for sub in one_word_subs:
                new_row = row.to_dict()
                new_row["clean_sentence"] = clean_sentence
                new_row["target_clean"] = target
                new_row["substitute"] = sub
                new_row["source_row_idx"] = idx
                rows.append(new_row)
        else:
            new_row = row.to_dict()
            new_row["clean_sentence"] = clean_sentence
            new_row["target_clean"] = target
            new_row[f"{args.sub_col}_oneword"] = one_word_subs
            new_row["first_substitute"] = one_word_subs[0]
            new_row["num_oneword_subs"] = len(one_word_subs)
            new_row["source_row_idx"] = idx
            rows.append(new_row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)

    print(f"Original rows: {len(df)}")
    print(f"Filtered rows: {len(out_df)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()