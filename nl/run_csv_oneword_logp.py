import argparse
import json
import math
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


WORD_RE = re.compile(r"^[A-Za-z]+$")


def is_single_word(x: str) -> bool:
    if not isinstance(x, str):
        return False
    x = x.strip()
    return bool(WORD_RE.fullmatch(x))


def is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x)


def safe_str(x) -> str:
    if is_missing(x):
        return ""
    return str(x)


def match_case_like(word: str, ref: str) -> str:
    if not word:
        return word
    if ref.isupper():
        return word.upper()
    if ref[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def find_marked_target(sentence: str):
    """
    Handles ProLex-style sentence with **target**.
    Returns clean_sentence, prefix, target, suffix.
    """
    m = re.search(r"\*\*(.+?)\*\*", sentence)
    if not m:
        return None

    prefix = sentence[:m.start()]
    target = m.group(1)
    suffix = sentence[m.end():]
    clean_sentence = prefix + target + suffix

    return clean_sentence, prefix, target, suffix


def find_target_in_sentence(sentence: str, target: str, require_unique=True):
    """
    Fallback if prefix/suffix are not already in the CSV.
    Finds target in clean_sentence.
    """
    if not sentence or not target:
        return None

    # First try exact match.
    matches = list(re.finditer(re.escape(target), sentence))

    # If exact match fails, try case-insensitive word match.
    if len(matches) == 0:
        pattern = re.compile(r"\b" + re.escape(target) + r"\b", re.IGNORECASE)
        matches = list(pattern.finditer(sentence))

    if len(matches) == 0:
        return None

    if require_unique and len(matches) != 1:
        return "REPEATED"

    m = matches[0]
    actual_target = sentence[m.start():m.end()]
    prefix = sentence[:m.start()]
    suffix = sentence[m.end():]

    return sentence, prefix, actual_target, suffix


def build_example_from_row(row, args, row_idx):
    """
    Expected cleaned CSV columns:
      clean_sentence
      target_clean
      substitute

    Optional:
      prefix
      suffix
      Sentence with **target** markers
    """
    target = safe_str(row.get(args.target_col)).strip()
    substitute = safe_str(row.get(args.sub_col)).strip()

    if not target or not substitute:
        return None, "missing_target_or_substitute"

    # 1. Prefer prefix/suffix if available, e.g. filtered SWORDS CSV.
    if args.prefix_col in row and args.suffix_col in row:
        prefix = safe_str(row.get(args.prefix_col))
        suffix = safe_str(row.get(args.suffix_col))

        if prefix or suffix:
            sentence = safe_str(row.get(args.sentence_col))
            if not sentence:
                sentence = prefix + target + suffix

            return {
                "row_idx": row_idx,
                "dataset": args.dataset_label,
                "sentence": sentence,
                "prefix": prefix,
                "suffix": suffix,
                "target": target,
                "substitute": substitute,
            }, None

    # 2. Use ProLex original marked sentence if available.
    if args.marked_sentence_col in row:
        marked_sentence = safe_str(row.get(args.marked_sentence_col))
        extracted = find_marked_target(marked_sentence)
        if extracted is not None:
            clean_sentence, prefix, marked_target, suffix = extracted
            return {
                "row_idx": row_idx,
                "dataset": args.dataset_label,
                "sentence": clean_sentence,
                "prefix": prefix,
                "suffix": suffix,
                "target": marked_target,
                "substitute": substitute,
            }, None

    # 3. Fallback: find target in clean_sentence.
    sentence = safe_str(row.get(args.sentence_col))
    found = find_target_in_sentence(
        sentence,
        target,
        require_unique=not args.allow_repeated_target,
    )

    if found is None:
        return None, "target_not_found"

    if found == "REPEATED":
        return None, "target_repeated"

    clean_sentence, prefix, actual_target, suffix = found

    return {
        "row_idx": row_idx,
        "dataset": args.dataset_label,
        "sentence": clean_sentence,
        "prefix": prefix,
        "suffix": suffix,
        "target": actual_target,
        "substitute": substitute,
    }, None


def load_model_and_tokenizer(
    model_name,
    dtype="bf16",
    trust_remote_code=False,
    device_map_auto=False,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )

    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "This script needs a fast tokenizer because it uses offset_mapping. "
            "Try another tokenizer/model or make sure use_fast=True works."
        )

    torch_dtype = torch.float32
    if torch.cuda.is_available():
        if dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "fp32":
            torch_dtype = torch.float32

    kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype,
    }

    if device_map_auto:
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if not device_map_auto:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

    model.eval()
    return model, tokenizer


def token_indices_for_continuation(tokenizer, text, start, end):
    """
    Tokenize full text and find token indices overlapping the continuation span.

    This is more robust than separately tokenizing prefix and prefix+word,
    because BPE/SentencePiece often merges the preceding space with the word.
    """
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    cont_indices = []

    for i, (a, b) in enumerate(offsets):
        if b <= start or a >= end:
            continue

        # Allow token to include whitespace just before/after the word.
        if a < start and text[a:start].strip():
            return None, None, "token_crosses_left_nonspace"

        if b > end and text[end:b].strip():
            return None, None, "token_crosses_right_nonspace"

        cont_indices.append(i)

    if not cont_indices:
        return None, None, "no_continuation_tokens"

    cont_ids = [input_ids[i] for i in cont_indices]
    return input_ids, cont_indices, None


@torch.no_grad()
def continuation_logprob(model, tokenizer, prefix, continuation, require_single_token=True):
    text = prefix + continuation
    start = len(prefix)
    end = len(text)

    input_ids, cont_indices, err = token_indices_for_continuation(
        tokenizer,
        text,
        start,
        end,
    )

    if err is not None:
        return None, None, None, err

    cont_ids = [input_ids[i] for i in cont_indices]

    if require_single_token and len(cont_ids) != 1:
        return None, cont_ids, None, "not_one_token"

    max_pos = getattr(model.config, "max_position_embeddings", None)
    if max_pos is not None and len(input_ids) > max_pos:
        return None, cont_ids, None, "too_long"

    device = model.get_input_embeddings().weight.device
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    out = model(input_ids=input_tensor)
    log_probs = torch.log_softmax(out.logits, dim=-1)

    total_logp = 0.0

    for tok_idx in cont_indices:
        if tok_idx == 0:
            return None, cont_ids, None, "no_context_for_first_token"

        tok_id = input_ids[tok_idx]
        total_logp += log_probs[0, tok_idx - 1, tok_id].item()

    tokens = tokenizer.convert_ids_to_tokens(cont_ids)
    return total_logp, cont_ids, tokens, None


def summarize_df(df):
    if len(df) == 0:
        return {"n": 0}

    scores = df["score"].astype(float).to_numpy()
    orig = df["orig_logp"].astype(float).to_numpy()
    sub = df["sub_logp"].astype(float).to_numpy()
    positive = scores[scores > 0]

    return {
        "n": int(len(scores)),
        "mean_score": float(scores.mean()),
        "median_score": float(np.median(scores)),
        "std_score": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "min_score": float(scores.min()),
        "max_score": float(scores.max()),
        "frac_positive": float((scores > 0).mean()),
        "positive_half_mean": float(positive.mean()) if len(positive) else None,
        "mean_orig_logp": float(orig.mean()),
        "mean_sub_logp": float(sub.mean()),
        "mean_orig_nll": float((-orig).mean()),
        "mean_sub_nll": float((-sub).mean()),
    }


def run_experiment(args):
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        device_map_auto=args.device_map_auto,
    )

    df = pd.read_csv(args.data)
    rows = []
    skip = Counter()

    require_single_token = not args.no_single_token_filter

    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc=f"running {args.dataset_label}"):
        ex, err = build_example_from_row(row, args, row_idx)

        if err is not None:
            skip[err] += 1
            continue

        target = ex["target"].strip()
        substitute = ex["substitute"].strip()

        if args.match_case:
            substitute = match_case_like(substitute, target)

        if not is_single_word(target):
            skip["target_not_single_word"] += 1
            continue

        if not is_single_word(substitute):
            skip["sub_not_single_word"] += 1
            continue

        if target.lower() == substitute.lower():
            skip["same_as_target"] += 1
            continue

        prefix = ex["prefix"]
        suffix = ex["suffix"]

        orig_logp, orig_ids, orig_tokens, err = continuation_logprob(
            model,
            tokenizer,
            prefix,
            target,
            require_single_token=require_single_token,
        )
        if err is not None:
            skip[f"orig_{err}"] += 1
            continue

        sub_logp, sub_ids, sub_tokens, err = continuation_logprob(
            model,
            tokenizer,
            prefix,
            substitute,
            require_single_token=require_single_token,
        )
        if err is not None:
            skip[f"sub_{err}"] += 1
            continue

        score = orig_logp - sub_logp

        rows.append({
            "row_idx": ex["row_idx"],
            "dataset": ex["dataset"],
            "sentence": ex["sentence"],
            "changed_sentence": prefix + substitute + suffix,
            "prefix": prefix,
            "suffix": suffix,
            "target": target,
            "substitute": substitute,
            "orig_logp": orig_logp,
            "sub_logp": sub_logp,
            "orig_nll": -orig_logp,
            "sub_nll": -sub_logp,
            "score": score,
            "orig_token_ids": json.dumps(orig_ids),
            "sub_token_ids": json.dumps(sub_ids),
            "orig_tokens": json.dumps(orig_tokens, ensure_ascii=False),
            "sub_tokens": json.dumps(sub_tokens, ensure_ascii=False),
            "orig_num_tokens": len(orig_ids),
            "sub_num_tokens": len(sub_ids),
        })

        if args.max_examples is not None and len(rows) >= args.max_examples:
            break

    out_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)

    summary = summarize_df(out_df)
    summary["skip_counts"] = dict(skip)
    summary["input_rows"] = int(len(df))
    summary["kept_rows"] = int(len(out_df))
    summary["model"] = args.model
    summary["dataset_label"] = args.dataset_label
    summary["data"] = args.data

    summary_path = args.out.replace(".csv", ".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSaved results:", args.out)
    print("Saved summary:", summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def average_ranks(x):
    x = np.asarray(x)
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)

    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1

        avg_rank = (i + 1 + j + 1) / 2.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    return ranks


def auc_old_higher(old_scores, new_scores):
    old_scores = np.asarray(old_scores, dtype=float)
    new_scores = np.asarray(new_scores, dtype=float)

    if len(old_scores) == 0 or len(new_scores) == 0:
        return None

    scores = np.concatenate([old_scores, new_scores])
    labels = np.concatenate([
        np.ones(len(old_scores)),
        np.zeros(len(new_scores)),
    ])

    ranks = average_ranks(scores)
    n_pos = len(old_scores)
    n_neg = len(new_scores)

    rank_sum_pos = ranks[labels == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def cohens_d(old_scores, new_scores):
    old_scores = np.asarray(old_scores, dtype=float)
    new_scores = np.asarray(new_scores, dtype=float)

    if len(old_scores) < 2 or len(new_scores) < 2:
        return None

    n1, n2 = len(old_scores), len(new_scores)
    s1 = old_scores.std(ddof=1)
    s2 = new_scores.std(ddof=1)

    denom = n1 + n2 - 2
    if denom <= 0:
        return None

    pooled = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / denom)

    if pooled == 0:
        return None

    return float((old_scores.mean() - new_scores.mean()) / pooled)


def compare_results(args):
    old_df = pd.read_csv(args.old_csv)
    new_df = pd.read_csv(args.new_csv)

    old_scores = old_df["score"].astype(float).to_numpy()
    new_scores = new_df["score"].astype(float).to_numpy()

    out = {
        "old_file": args.old_csv,
        "new_file": args.new_csv,
        "old": summarize_df(old_df),
        "new": summarize_df(new_df),
        "difference_mean_old_minus_new": float(old_scores.mean() - new_scores.mean()),
        "auc_old_higher_than_new": auc_old_higher(old_scores, new_scores),
        "cohens_d_old_minus_new": cohens_d(old_scores, new_scores),
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))

    if args.compare_out:
        with open(args.compare_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("Saved compare summary:", args.compare_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run", "compare"], default="run")

    # Run mode
    parser.add_argument("--data", help="Cleaned CSV input")
    parser.add_argument("--out", help="Result CSV output")
    parser.add_argument("--dataset_label", default="dataset")

    parser.add_argument("--model", default="allenai/OLMo-1B-hf")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--device_map_auto", action="store_true")

    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--no_single_token_filter", action="store_true")
    parser.add_argument("--allow_repeated_target", action="store_true")
    parser.add_argument("--no_match_case", action="store_true")

    # CSV column names
    parser.add_argument("--sentence_col", default="clean_sentence")
    parser.add_argument("--target_col", default="target_clean")
    parser.add_argument("--sub_col", default="substitute")
    parser.add_argument("--prefix_col", default="prefix")
    parser.add_argument("--suffix_col", default="suffix")
    parser.add_argument("--marked_sentence_col", default="Sentence")

    # Compare mode
    parser.add_argument("--old_csv")
    parser.add_argument("--new_csv")
    parser.add_argument("--compare_out", default=None)

    args = parser.parse_args()
    args.match_case = not args.no_match_case

    if args.mode == "run":
        if not args.data or not args.out:
            raise ValueError("--data and --out are required in run mode")
        run_experiment(args)

    else:
        if not args.old_csv or not args.new_csv:
            raise ValueError("--old_csv and --new_csv are required in compare mode")
        compare_results(args)


if __name__ == "__main__":
    main()