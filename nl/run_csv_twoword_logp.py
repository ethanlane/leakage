#!/usr/bin/env python3
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


def safe_str(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def is_single_word(x):
    return isinstance(x, str) and bool(WORD_RE.fullmatch(x.strip()))


def match_case_like(word, ref):
    if not word:
        return word
    if ref.isupper():
        return word.upper()
    if ref[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def find_col(df, preferred, fallbacks=None):
    fallbacks = fallbacks or []
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    for c in [preferred] + fallbacks:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]

    return None


def locate_span(
    row,
    text,
    word,
    offset_col=None,
    end_col=None,
    prefix_col=None,
    require_unique=True,
):
    word = safe_str(word).strip()
    if not word:
        return None, "empty_word"

    # 1. Explicit offset/end columns.
    if offset_col and offset_col in row and pd.notna(row[offset_col]):
        start = int(row[offset_col])

        if end_col and end_col in row and pd.notna(row[end_col]):
            end = int(row[end_col])
        else:
            end = start + len(word)

        if 0 <= start < end <= len(text) and text[start:end].lower() == word.lower():
            return (start, end, text[start:end]), None

    # 2. Prefix column, useful for target1 from older one-word CSVs.
    if prefix_col and prefix_col in row and pd.notna(row[prefix_col]):
        prefix = safe_str(row[prefix_col])
        start = len(prefix)
        end = start + len(word)

        if 0 <= start < end <= len(text) and text[start:end].lower() == word.lower():
            return (start, end, text[start:end]), None

    # 3. Unique exact/case-insensitive match fallback.
    matches = list(re.finditer(re.escape(word), text))
    if not matches:
        matches = list(re.finditer(re.escape(word), text, flags=re.IGNORECASE))

    if not matches:
        return None, "target_not_found"

    if require_unique and len(matches) != 1:
        return None, "target_repeated"

    m = matches[0]
    return (m.start(), m.end(), text[m.start():m.end()]), None


def spans_overlap(a, b):
    return not (a[1] <= b[0] or b[1] <= a[0])


def replace_two_spans_with_mapping(text, span1, sub1, span2, sub2):
    """
    Replace two spans and return:
      changed_text,
      substitute spans in changed_text,
      error

    span = (start, end, original_surface)
    """
    items = [
        {"name": "sub1", "start": span1[0], "end": span1[1], "sub": sub1},
        {"name": "sub2", "start": span2[0], "end": span2[1], "sub": sub2},
    ]
    items = sorted(items, key=lambda x: x["start"])

    if items[0]["end"] > items[1]["start"]:
        return None, None, "overlapping_spans"

    out_parts = []
    new_spans = {}
    cursor = 0
    new_len = 0

    for it in items:
        before = text[cursor:it["start"]]
        out_parts.append(before)
        new_len += len(before)

        new_start = new_len
        out_parts.append(it["sub"])
        new_len += len(it["sub"])
        new_end = new_len

        new_spans[it["name"]] = (new_start, new_end)
        cursor = it["end"]

    out_parts.append(text[cursor:])
    changed_text = "".join(out_parts)

    return changed_text, new_spans, None


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
        raise ValueError("Need a fast tokenizer for offset_mapping.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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


@torch.no_grad()
def forward_text(model, tokenizer, text):
    device = model.get_input_embeddings().weight.device

    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    if len(input_ids) < 2:
        return None, "too_short"

    max_pos = getattr(model.config, "max_position_embeddings", None)
    if max_pos is not None and len(input_ids) > max_pos:
        return None, "too_long"

    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    out = model(input_ids=x)
    log_probs = torch.log_softmax(out.logits, dim=-1)

    return {
        "input_ids": input_ids,
        "offsets": offsets,
        "log_probs": log_probs,
    }, None


def span_logprob_from_forward(
    fwd,
    text,
    start,
    end,
    require_single_token=True,
):
    input_ids = fwd["input_ids"]
    offsets = fwd["offsets"]
    log_probs = fwd["log_probs"]

    indices = []

    for i, (a, b) in enumerate(offsets):
        if b <= start or a >= end:
            continue

        # Allow token to include whitespace outside the span,
        # but not punctuation/letters outside the target span.
        if a < start and text[a:start].strip():
            return None, None, None, "token_crosses_left_nonspace"

        if b > end and text[end:b].strip():
            return None, None, None, "token_crosses_right_nonspace"

        indices.append(i)

    if not indices:
        return None, None, None, "no_span_tokens"

    ids = [input_ids[i] for i in indices]

    if require_single_token and len(ids) != 1:
        return None, ids, None, "not_one_token"

    total_logp = 0.0

    for i in indices:
        if i == 0:
            return None, ids, None, "no_context_for_first_token"

        tok_id = input_ids[i]
        total_logp += log_probs[0, i - 1, tok_id].item()

    return total_logp, ids, indices, None


@torch.no_grad()
def continuation_logprob(
    model,
    tokenizer,
    prefix,
    word,
    require_single_token=True,
):
    """
    Scores log P(word | prefix) using only prefix + word.
    This is used for independent original-prefix substitute scoring.
    """
    text = prefix + word
    start = len(prefix)
    end = len(text)

    fwd, err = forward_text(model, tokenizer, text)
    if err is not None:
        return None, None, None, err

    lp, ids, indices, err = span_logprob_from_forward(
        fwd,
        text,
        start,
        end,
        require_single_token=require_single_token,
    )

    tokens = tokenizer.convert_ids_to_tokens(ids) if ids is not None else None
    return lp, ids, tokens, err


def mean_nll_from_forward(fwd):
    """
    Average per-token NLL over the whole sentence.
    First token is not scored because causal LM has no prior context.
    """
    input_ids = fwd["input_ids"]
    log_probs = fwd["log_probs"]

    total_nll = 0.0
    count = 0

    for i in range(1, len(input_ids)):
        tok_id = input_ids[i]
        total_nll += -log_probs[0, i - 1, tok_id].item()
        count += 1

    mean_nll = total_nll / max(count, 1)
    return mean_nll, total_nll, count


def summarize_df(df, score_col):
    if len(df) == 0:
        return {"n": 0, "score_col": score_col}

    scores = df[score_col].astype(float).to_numpy()
    pos = scores[scores > 0]

    out = {
        "n": int(len(scores)),
        "score_col": score_col,
        "mean_score": float(scores.mean()),
        "median_score": float(np.median(scores)),
        "std_score": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "min_score": float(scores.min()),
        "max_score": float(scores.max()),
        "frac_positive": float((scores > 0).mean()),
        "positive_half_mean": float(pos.mean()) if len(pos) else None,
    }

    for c in [
        "orig_two_logp",
        "sub_two_independent_logp",
        "sub_two_sequential_logp",
        "orig_sentence_avg_logp",
        "sub_sentence_avg_logp",
        "orig_context_mean_nll",
        "sub_context_mean_nll",
    ]:
        if c in df.columns:
            out[f"mean_{c}"] = float(df[c].astype(float).mean())

    return out


def run(args):
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        device_map_auto=args.device_map_auto,
    )

    df = pd.read_csv(args.input)

    sentence_col = find_col(df, args.sentence_col, ["clean_sentence", "sentence"])
    target1_col = find_col(df, args.target1_col, ["target1", "target_clean", "target"])
    sub1_col = find_col(df, args.sub1_col, ["substitute1", "substitute"])
    target2_col = find_col(df, args.target2_col, ["target2"])
    sub2_col = find_col(df, args.sub2_col, ["substitute2", "candidate2", "chosen_substitute2"])

    required = {
        "sentence": sentence_col,
        "target1": target1_col,
        "substitute1": sub1_col,
        "target2": target2_col,
        "substitute2": sub2_col,
    }

    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(
            f"Missing required columns for {missing}. Existing columns: {list(df.columns)}"
        )

    require_single_token = not args.no_single_token_filter

    rows = []
    skip = Counter()

    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc=f"running {args.dataset_label}"):
        sentence = safe_str(row[sentence_col])

        target1 = safe_str(row[target1_col]).strip()
        sub1 = safe_str(row[sub1_col]).strip()
        target2 = safe_str(row[target2_col]).strip()
        sub2 = safe_str(row[sub2_col]).strip()

        if not all([sentence, target1, sub1, target2, sub2]):
            skip["missing_fields"] += 1
            continue

        if args.require_alpha_words:
            bad = False
            for name, w in [
                ("target1", target1),
                ("sub1", sub1),
                ("target2", target2),
                ("sub2", sub2),
            ]:
                if not is_single_word(w):
                    skip[f"{name}_not_single_word"] += 1
                    bad = True
                    break
            if bad:
                continue

        if target1.lower() == sub1.lower() or target2.lower() == sub2.lower():
            skip["same_as_target"] += 1
            continue

        span1, err = locate_span(
            row,
            sentence,
            target1,
            offset_col=args.target1_offset_col,
            end_col=args.target1_end_col,
            prefix_col=args.prefix_col,
            require_unique=not args.allow_repeated_target,
        )
        if err is not None:
            skip[f"target1_{err}"] += 1
            continue

        span2, err = locate_span(
            row,
            sentence,
            target2,
            offset_col=args.target2_offset_col,
            end_col=args.target2_end_col,
            prefix_col=None,
            require_unique=not args.allow_repeated_target,
        )
        if err is not None:
            skip[f"target2_{err}"] += 1
            continue

        if spans_overlap(span1, span2):
            skip["target_spans_overlap"] += 1
            continue

        target1_surface = span1[2]
        target2_surface = span2[2]

        if args.match_case:
            sub1 = match_case_like(sub1, target1_surface)
            sub2 = match_case_like(sub2, target2_surface)

        changed_sentence, sub_spans, err = replace_two_spans_with_mapping(
            sentence,
            span1,
            sub1,
            span2,
            sub2,
        )
        if err is not None:
            skip[err] += 1
            continue

        # ------------------------------------------------------------
        # Original sentence forward pass.
        # ------------------------------------------------------------
        orig_fwd, err = forward_text(model, tokenizer, sentence)
        if err is not None:
            skip[f"orig_{err}"] += 1
            continue

        orig1_logp, orig1_ids, _, err = span_logprob_from_forward(
            orig_fwd,
            sentence,
            span1[0],
            span1[1],
            require_single_token=require_single_token,
        )
        if err is not None:
            skip[f"orig1_{err}"] += 1
            continue

        orig2_logp, orig2_ids, _, err = span_logprob_from_forward(
            orig_fwd,
            sentence,
            span2[0],
            span2[1],
            require_single_token=require_single_token,
        )
        if err is not None:
            skip[f"orig2_{err}"] += 1
            continue

        orig_context_mean_nll, orig_context_sum_nll, orig_context_num_tokens = mean_nll_from_forward(orig_fwd)
        orig_sentence_avg_logp = -orig_context_mean_nll

        # ------------------------------------------------------------
        # Fully substituted sentence forward pass.
        # Used for sequential changed-token score and whole-sentence score.
        # ------------------------------------------------------------
        changed_fwd, err = forward_text(model, tokenizer, changed_sentence)
        if err is not None:
            skip[f"changed_{err}"] += 1
            continue

        seq_sub1_start, seq_sub1_end = sub_spans["sub1"]
        seq_sub2_start, seq_sub2_end = sub_spans["sub2"]

        seq_sub1_logp, seq_sub1_ids, _, err = span_logprob_from_forward(
            changed_fwd,
            changed_sentence,
            seq_sub1_start,
            seq_sub1_end,
            require_single_token=require_single_token,
        )
        if err is not None:
            skip[f"seq_sub1_{err}"] += 1
            continue

        seq_sub2_logp, seq_sub2_ids, _, err = span_logprob_from_forward(
            changed_fwd,
            changed_sentence,
            seq_sub2_start,
            seq_sub2_end,
            require_single_token=require_single_token,
        )
        if err is not None:
            skip[f"seq_sub2_{err}"] += 1
            continue

        sub_context_mean_nll, sub_context_sum_nll, sub_context_num_tokens = mean_nll_from_forward(changed_fwd)
        sub_sentence_avg_logp = -sub_context_mean_nll

        # ------------------------------------------------------------
        # Independent substitute scoring.
        # Both substitutes are scored under the original prefix.
        # ------------------------------------------------------------
        prefix1 = sentence[:span1[0]]
        prefix2 = sentence[:span2[0]]

        ind_sub1_logp, ind_sub1_ids, ind_sub1_tokens, err = continuation_logprob(
            model,
            tokenizer,
            prefix1,
            sub1,
            require_single_token=require_single_token,
        )
        if err is not None:
            skip[f"ind_sub1_{err}"] += 1
            continue

        ind_sub2_logp, ind_sub2_ids, ind_sub2_tokens, err = continuation_logprob(
            model,
            tokenizer,
            prefix2,
            sub2,
            require_single_token=require_single_token,
        )
        if err is not None:
            skip[f"ind_sub2_{err}"] += 1
            continue

        # ------------------------------------------------------------
        # Three main scores.
        # ------------------------------------------------------------

        # Original two changed words in original sentence.
        orig_two_logp = orig1_logp + orig2_logp

        # Method 1: independent original-prefix substitute score.
        sub_two_independent_logp = ind_sub1_logp + ind_sub2_logp
        independent_score = orig_two_logp - sub_two_independent_logp

        # Method 2: sequential fully substituted context score.
        sub_two_sequential_logp = seq_sub1_logp + seq_sub2_logp
        sequential_score = orig_two_logp - sub_two_sequential_logp

        # Method 3: whole sentence average log-probability score.
        whole_sentence_avg_logp_score = orig_sentence_avg_logp - sub_sentence_avg_logp

        # Same as whole_sentence_avg_logp_score, but expressed as NLL difference.
        # Positive means original sentence has lower average NLL.
        context_score = sub_context_mean_nll - orig_context_mean_nll

        rows.append({
            "row_idx": row_idx,
            "dataset": args.dataset_label,

            "sentence": sentence,
            "changed_sentence_two": changed_sentence,

            "target1": target1_surface,
            "substitute1": sub1,
            "target1_offset": span1[0],
            "target1_end": span1[1],

            "target2": target2_surface,
            "substitute2": sub2,
            "target2_offset": span2[0],
            "target2_end": span2[1],

            # Original changed-word logprobs.
            "orig1_logp": orig1_logp,
            "orig2_logp": orig2_logp,
            "orig_two_logp": orig_two_logp,

            # Independent substitute logprobs.
            "ind_sub1_logp": ind_sub1_logp,
            "ind_sub2_logp": ind_sub2_logp,
            "sub_two_independent_logp": sub_two_independent_logp,
            "independent_score": independent_score,

            # Sequential substitute logprobs.
            "seq_sub1_logp": seq_sub1_logp,
            "seq_sub2_logp": seq_sub2_logp,
            "sub_two_sequential_logp": sub_two_sequential_logp,
            "sequential_score": sequential_score,

            # Whole sentence average NLL/logprob.
            "orig_context_mean_nll": orig_context_mean_nll,
            "sub_context_mean_nll": sub_context_mean_nll,
            "orig_context_sum_nll": orig_context_sum_nll,
            "sub_context_sum_nll": sub_context_sum_nll,
            "orig_context_num_tokens": orig_context_num_tokens,
            "sub_context_num_tokens": sub_context_num_tokens,
            "context_score": context_score,

            "orig_sentence_avg_logp": orig_sentence_avg_logp,
            "sub_sentence_avg_logp": sub_sentence_avg_logp,
            "whole_sentence_avg_logp_score": whole_sentence_avg_logp_score,

            # Token info.
            "orig1_token_ids": json.dumps(orig1_ids),
            "orig2_token_ids": json.dumps(orig2_ids),
            "ind_sub1_token_ids": json.dumps(ind_sub1_ids),
            "ind_sub2_token_ids": json.dumps(ind_sub2_ids),
            "seq_sub1_token_ids": json.dumps(seq_sub1_ids),
            "seq_sub2_token_ids": json.dumps(seq_sub2_ids),

            "orig1_tokens": json.dumps(tokenizer.convert_ids_to_tokens(orig1_ids), ensure_ascii=False),
            "orig2_tokens": json.dumps(tokenizer.convert_ids_to_tokens(orig2_ids), ensure_ascii=False),
            "ind_sub1_tokens": json.dumps(ind_sub1_tokens, ensure_ascii=False),
            "ind_sub2_tokens": json.dumps(ind_sub2_tokens, ensure_ascii=False),
            "seq_sub1_tokens": json.dumps(tokenizer.convert_ids_to_tokens(seq_sub1_ids), ensure_ascii=False),
            "seq_sub2_tokens": json.dumps(tokenizer.convert_ids_to_tokens(seq_sub2_ids), ensure_ascii=False),
        })

        if args.max_examples is not None and len(rows) >= args.max_examples:
            break

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)

    summary = {
        "input_rows": int(len(df)),
        "kept_rows": int(len(out_df)),
        "skip_counts": dict(skip),
        "model": args.model,
        "dataset_label": args.dataset_label,
        "main_independent": summarize_df(out_df, "independent_score"),
        "sequential": summarize_df(out_df, "sequential_score"),
        "whole_sentence_avg_logp": summarize_df(out_df, "whole_sentence_avg_logp_score"),
        "context_nll_equivalent": summarize_df(out_df, "context_score"),
    }

    summary_path = args.out.replace(".csv", ".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSaved:", args.out)
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

        avg = (i + 1 + j + 1) / 2.0
        ranks[order[i:j + 1]] = avg
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

    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def cohens_d(old_scores, new_scores):
    old_scores = np.asarray(old_scores, dtype=float)
    new_scores = np.asarray(new_scores, dtype=float)

    if len(old_scores) < 2 or len(new_scores) < 2:
        return None

    n1, n2 = len(old_scores), len(new_scores)
    s1 = old_scores.std(ddof=1)
    s2 = new_scores.std(ddof=1)

    pooled = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

    if pooled == 0:
        return None

    return float((old_scores.mean() - new_scores.mean()) / pooled)


def compare(args):
    old_df = pd.read_csv(args.old_csv)
    new_df = pd.read_csv(args.new_csv)

    score_col = args.score_col

    if score_col not in old_df.columns:
        raise ValueError(f"Missing {score_col} in old CSV. Existing: {list(old_df.columns)}")
    if score_col not in new_df.columns:
        raise ValueError(f"Missing {score_col} in new CSV. Existing: {list(new_df.columns)}")

    old_scores = old_df[score_col].astype(float).to_numpy()
    new_scores = new_df[score_col].astype(float).to_numpy()

    out = {
        "score_col": score_col,
        "old_file": args.old_csv,
        "new_file": args.new_csv,
        "old": summarize_df(old_df, score_col),
        "new": summarize_df(new_df, score_col),
        "difference_mean_old_minus_new": float(old_scores.mean() - new_scores.mean()),
        "auc_old_higher_than_new": auc_old_higher(old_scores, new_scores),
        "cohens_d_old_minus_new": cohens_d(old_scores, new_scores),
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))

    if args.compare_out:
        os.makedirs(os.path.dirname(args.compare_out) or ".", exist_ok=True)
        with open(args.compare_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("Saved compare summary:", args.compare_out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["run", "compare"], default="run")

    # Run mode.
    p.add_argument("--input")
    p.add_argument("--out")
    p.add_argument("--dataset_label", default="dataset")

    p.add_argument("--model", default="allenai/OLMo-1B-hf")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--device_map_auto", action="store_true")

    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--no_single_token_filter", action="store_true")
    p.add_argument("--allow_repeated_target", action="store_true")
    p.add_argument("--no_match_case", action="store_true")
    p.add_argument("--require_alpha_words", action="store_true", default=True)

    # Column names.
    p.add_argument("--sentence_col", default="clean_sentence")
    p.add_argument("--target1_col", default="target1")
    p.add_argument("--sub1_col", default="substitute1")
    p.add_argument("--target2_col", default="target2")
    p.add_argument("--sub2_col", default="substitute2")

    p.add_argument("--target1_offset_col", default="target1_offset")
    p.add_argument("--target1_end_col", default="target1_end")
    p.add_argument("--target2_offset_col", default="target2_offset")
    p.add_argument("--target2_end_col", default="target2_end")
    p.add_argument("--prefix_col", default="prefix")

    # Compare mode.
    p.add_argument("--old_csv")
    p.add_argument("--new_csv")
    p.add_argument("--score_col", default="independent_score")
    p.add_argument("--compare_out")

    args = p.parse_args()
    args.match_case = not args.no_match_case

    if args.mode == "run":
        if not args.input or not args.out:
            raise ValueError("--input and --out are required in run mode")
        run(args)
    else:
        if not args.old_csv or not args.new_csv:
            raise ValueError("--old_csv and --new_csv are required in compare mode")
        compare(args)


if __name__ == "__main__":
    main()