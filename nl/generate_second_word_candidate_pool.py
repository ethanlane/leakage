#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import Counter

import pandas as pd
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.tokenize import TreebankWordTokenizer


WORD_RE = re.compile(r"^[A-Za-z]+$")

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "so",
    "of", "to", "in", "on", "at", "by", "for", "from", "with", "about",
    "as", "into", "like", "through", "after", "over", "between", "out",
    "against", "during", "without", "before", "under", "around", "among",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "done", "have", "has", "had",
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those",
    "there", "here", "not", "no", "yes",
}


def ensure_nltk():
    for pkg in [
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    ]:
        try:
            nltk.data.find(pkg)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


def is_one_word(x):
    return isinstance(x, str) and bool(WORD_RE.fullmatch(x.strip()))


def penn_to_wordnet_pos(tag):
    if tag.startswith("N"):
        return wn.NOUN
    if tag.startswith("V"):
        return wn.VERB
    if tag.startswith("J"):
        return wn.ADJ
    if tag.startswith("R"):
        return wn.ADV
    return None


def match_case_like(word, ref):
    if ref.isupper():
        return word.upper()
    if ref[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def simple_inflect(candidate, original, penn_tag):
    """
    Simple automatic morphology matching.
    Not perfect, but okay before LLM validation.
    """
    c = candidate.lower()

    if penn_tag in {"NNS", "NNPS"}:
        if not c.endswith("s"):
            if c.endswith("y") and len(c) > 1 and c[-2] not in "aeiou":
                c = c[:-1] + "ies"
            elif c.endswith(("s", "x", "z", "ch", "sh")):
                c = c + "es"
            else:
                c = c + "s"

    elif penn_tag in {"VBD", "VBN"}:
        if not c.endswith("ed"):
            if c.endswith("e"):
                c = c + "d"
            else:
                c = c + "ed"

    elif penn_tag == "VBG":
        if c.endswith("e") and len(c) > 2:
            c = c[:-1] + "ing"
        elif not c.endswith("ing"):
            c = c + "ing"

    elif penn_tag == "VBZ":
        if not c.endswith("s"):
            if c.endswith("y") and len(c) > 1 and c[-2] not in "aeiou":
                c = c[:-1] + "ies"
            elif c.endswith(("s", "x", "z", "ch", "sh")):
                c = c + "es"
            else:
                c = c + "s"

    return match_case_like(c, original)


def get_context_free_candidates(word, wn_pos, penn_tag, max_candidates=30):
    """
    Context-free thesaurus-like candidate generation.

    Uses all WordNet synsets for the word with the same coarse POS.
    Does NOT look at sentence context.
    """
    scored = {}

    for syn in wn.synsets(word.lower(), pos=wn_pos):
        for lemma in syn.lemmas():
            raw = lemma.name().replace("_", " ").strip()

            if " " in raw or "-" in raw:
                continue
            if not is_one_word(raw):
                continue
            if raw.lower() == word.lower():
                continue

            cand = simple_inflect(raw, word, penn_tag)

            if not is_one_word(cand):
                continue
            if cand.lower() == word.lower():
                continue

            # WordNet lemma count is weak frequency signal.
            scored[cand] = max(scored.get(cand, 0), lemma.count())

    ranked = sorted(scored.items(), key=lambda x: (-x[1], len(x[0]), x[0].lower()))
    return [c for c, _ in ranked[:max_candidates]]


def load_tokenizer(model_name, trust_remote_code=False):
    if not model_name:
        return None
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )


def is_one_model_token(tokenizer, word):
    if tokenizer is None:
        return True
    ids = tokenizer(" " + word, add_special_tokens=False).input_ids
    return len(ids) == 1


def find_first_target_offset(row, sentence, target):
    # 1. SWORDS-style explicit offset
    if "target_offset" in row and pd.notna(row["target_offset"]):
        off = int(row["target_offset"])
        if sentence[off:off + len(target)].lower() == target.lower():
            return off

    # 2. SWORDS-style prefix
    if "prefix" in row and pd.notna(row["prefix"]):
        off = len(str(row["prefix"]))
        if sentence[off:off + len(target)].lower() == target.lower():
            return off

    # 3. ProLex-style **target** marker
    if "Sentence" in row and pd.notna(row["Sentence"]):
        marked = str(row["Sentence"])
        m = re.search(r"\*\*(.+?)\*\*", marked)
        if m:
            marked_target = m.group(1)
            prefix_marked = marked[:m.start()]
            # remove any previous markers before the target, just in case
            prefix_clean = prefix_marked.replace("**", "")
            off = len(prefix_clean)

            if sentence[off:off + len(marked_target)].lower() == marked_target.lower():
                return off

    # 4. fallback: only accept unique target occurrence
    matches = list(re.finditer(re.escape(target), sentence, flags=re.IGNORECASE))
    if len(matches) == 1:
        return matches[0].start()

    return None


def build_target_options(sentence, first_start, first_end, tokenizer, args):
    tok = TreebankWordTokenizer()
    spans = list(tok.span_tokenize(sentence))
    tokens = [sentence[s:e] for s, e in spans]
    tags = pos_tag(tokens)

    options = []

    for rank, ((start, end), (word, tag)) in enumerate(zip(spans, tags)):
        lower = word.lower()

        # Do not pick original first target again.
        if not (end <= first_start or start >= first_end):
            continue

        if lower in STOPWORDS:
            continue
        if len(word) < args.min_word_len:
            continue
        if not is_one_word(word):
            continue

        wn_pos = penn_to_wordnet_pos(tag)
        if wn_pos is None:
            continue

        if args.require_one_token and not is_one_model_token(tokenizer, word):
            continue

        candidates = get_context_free_candidates(
            word,
            wn_pos,
            tag,
            max_candidates=args.max_candidates_per_target,
        )

        if args.require_one_token:
            candidates = [c for c in candidates if is_one_model_token(tokenizer, c)]

        # Remove duplicates case-insensitively.
        seen = set()
        deduped = []
        for c in candidates:
            key = c.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(c)

        if len(deduped) < args.min_candidates_per_target:
            continue

        options.append({
            "target2": word,
            "target2_offset": start,
            "target2_end": end,
            "target2_pos": tag,
            "target2_rank": rank,
            "candidates": deduped,
        })

        if len(options) >= args.max_target_options:
            break

    return options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--sentence_col", default="clean_sentence")
    parser.add_argument("--target_col", default="target_clean")
    parser.add_argument("--sub_col", default="substitute")
    parser.add_argument("--dataset_label", default="")

    parser.add_argument("--min_word_len", type=int, default=4)
    parser.add_argument("--max_target_options", type=int, default=5)
    parser.add_argument("--max_candidates_per_target", type=int, default=30)
    parser.add_argument("--min_candidates_per_target", type=int, default=1)

    parser.add_argument("--require_one_token", action="store_true")
    parser.add_argument("--tokenizer", default="allenai/OLMo-1B-hf")
    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument("--explode", action="store_true",
                        help="Output one row per target2-candidate pair.")
    args = parser.parse_args()

    ensure_nltk()

    tokenizer = None
    if args.require_one_token:
        tokenizer = load_tokenizer(args.tokenizer, args.trust_remote_code)

    df = pd.read_csv(args.input)
    rows = []
    skip = Counter()

    for idx, row in df.iterrows():
        sentence = str(row[args.sentence_col])
        target1 = str(row[args.target_col]).strip()
        substitute1 = str(row[args.sub_col]).strip()

        first_start = find_first_target_offset(row, sentence, target1)
        if first_start is None:
            skip["cannot_find_first_target"] += 1
            continue

        first_end = first_start + len(target1)

        options = build_target_options(
            sentence,
            first_start,
            first_end,
            tokenizer,
            args,
        )

        if not options:
            skip["no_target2_options"] += 1
            continue

        if args.explode:
            for opt in options:
                for cand in opt["candidates"]:
                    new_row = row.to_dict()
                    new_row["dataset"] = args.dataset_label
                    new_row["source_row_idx"] = idx
                    new_row["target1"] = target1
                    new_row["substitute1"] = substitute1
                    new_row["target1_offset"] = first_start
                    new_row["target1_end"] = first_end
                    new_row["target2"] = opt["target2"]
                    new_row["target2_offset"] = opt["target2_offset"]
                    new_row["target2_end"] = opt["target2_end"]
                    new_row["target2_pos"] = opt["target2_pos"]
                    new_row["candidate2"] = cand
                    rows.append(new_row)
        else:
            new_row = row.to_dict()
            new_row["dataset"] = args.dataset_label
            new_row["source_row_idx"] = idx
            new_row["target1"] = target1
            new_row["substitute1"] = substitute1
            new_row["target1_offset"] = first_start
            new_row["target1_end"] = first_end
            new_row["target2_options_json"] = json.dumps(options, ensure_ascii=False)
            new_row["num_target2_options"] = len(options)
            new_row["num_total_candidates2"] = sum(len(o["candidates"]) for o in options)
            rows.append(new_row)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(out_df)}")
    print(f"Saved to: {args.output}")
    print("Skip counts:")
    for k, v in sorted(skip.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
