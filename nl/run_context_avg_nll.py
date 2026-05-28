import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name, dtype="bf16", trust_remote_code=False, device_map_auto=False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )

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
def batch_mean_nll(model, tokenizer, texts, max_length=None):
    """
    Returns average per-token NLL for each full text.

    For causal LM:
      token t is predicted from tokens before t.
    So the first token is not scored.
    """
    device = model.get_input_embeddings().weight.device

    enc = tokenizer(
        texts,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        truncation=max_length is not None,
        max_length=max_length,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits

    # Predict token i+1 from position i.
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    token_nlls = -token_logps
    token_nlls = token_nlls * shift_mask

    sum_nll = token_nlls.sum(dim=1)
    num_tokens = shift_mask.sum(dim=1)

    mean_nll = sum_nll / num_tokens.clamp(min=1)

    return (
        mean_nll.detach().float().cpu().numpy(),
        sum_nll.detach().float().cpu().numpy(),
        num_tokens.detach().cpu().numpy(),
    )


def summarize(df, score_col="context_score"):
    if len(df) == 0:
        return {"n": 0}

    scores = df[score_col].astype(float).to_numpy()
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
        "mean_orig_context_nll": float(df["orig_context_mean_nll"].mean()),
        "mean_sub_context_nll": float(df["sub_context_mean_nll"].mean()),
    }


def run(args):
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        device_map_auto=args.device_map_auto,
    )

    df = pd.read_csv(args.input)

    if args.orig_col not in df.columns:
        raise ValueError(f"Missing original text column: {args.orig_col}")
    if args.sub_col not in df.columns:
        raise ValueError(f"Missing substituted text column: {args.sub_col}")

    orig_texts = df[args.orig_col].fillna("").astype(str).tolist()
    sub_texts = df[args.sub_col].fillna("").astype(str).tolist()

    orig_mean_all = []
    orig_sum_all = []
    orig_tok_all = []

    sub_mean_all = []
    sub_sum_all = []
    sub_tok_all = []

    for start in tqdm(range(0, len(df), args.batch_size), desc="scoring full contexts"):
        end = min(start + args.batch_size, len(df))

        orig_batch = orig_texts[start:end]
        sub_batch = sub_texts[start:end]

        orig_mean, orig_sum, orig_tok = batch_mean_nll(
            model, tokenizer, orig_batch, max_length=args.max_length
        )
        sub_mean, sub_sum, sub_tok = batch_mean_nll(
            model, tokenizer, sub_batch, max_length=args.max_length
        )

        orig_mean_all.extend(orig_mean.tolist())
        orig_sum_all.extend(orig_sum.tolist())
        orig_tok_all.extend(orig_tok.tolist())

        sub_mean_all.extend(sub_mean.tolist())
        sub_sum_all.extend(sub_sum.tolist())
        sub_tok_all.extend(sub_tok.tolist())

    out_df = df.copy()

    out_df["orig_context_mean_nll"] = orig_mean_all
    out_df["sub_context_mean_nll"] = sub_mean_all
    out_df["orig_context_sum_nll"] = orig_sum_all
    out_df["sub_context_sum_nll"] = sub_sum_all
    out_df["orig_context_num_tokens"] = orig_tok_all
    out_df["sub_context_num_tokens"] = sub_tok_all

    # Positive means original full context has lower average NLL.
    out_df["context_score"] = (
        out_df["sub_context_mean_nll"] - out_df["orig_context_mean_nll"]
    )

    # Equivalent log-prob view:
    # avg_logp = -mean_nll, so this equals orig_avg_logp - sub_avg_logp.
    out_df["orig_context_avg_logp"] = -out_df["orig_context_mean_nll"]
    out_df["sub_context_avg_logp"] = -out_df["sub_context_mean_nll"]
    out_df["context_logp_score"] = (
        out_df["orig_context_avg_logp"] - out_df["sub_context_avg_logp"]
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)

    summary = summarize(out_df)
    summary["model"] = args.model
    summary["input"] = args.input
    summary["output"] = args.out
    summary["orig_col"] = args.orig_col
    summary["sub_col"] = args.sub_col

    summary_path = args.out.replace(".csv", ".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:", args.out)
    print("Saved summary:", summary_path)
    print(json.dumps(summary, indent=2))


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

    n1, n2 = len(old_scores), len(new_scores)
    s1 = old_scores.std(ddof=1)
    s2 = new_scores.std(ddof=1)

    pooled = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))

    if pooled == 0:
        return None

    return float((old_scores.mean() - new_scores.mean()) / pooled)


def compare(args):
    old_df = pd.read_csv(args.old_csv)
    new_df = pd.read_csv(args.new_csv)

    old_scores = old_df["context_score"].astype(float).to_numpy()
    new_scores = new_df["context_score"].astype(float).to_numpy()

    out = {
        "old_file": args.old_csv,
        "new_file": args.new_csv,
        "old": summarize(old_df),
        "new": summarize(new_df),
        "difference_mean_old_minus_new": float(old_scores.mean() - new_scores.mean()),
        "auc_old_higher_than_new": auc_old_higher(old_scores, new_scores),
        "cohens_d_old_minus_new": cohens_d(old_scores, new_scores),
    }

    print(json.dumps(out, indent=2))

    if args.compare_out:
        with open(args.compare_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("Saved compare summary:", args.compare_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run", "compare"], default="run")

    parser.add_argument("--input", help="Previous result CSV with sentence and changed_sentence")
    parser.add_argument("--out", help="Output CSV with full-context scores")
    parser.add_argument("--orig_col", default="sentence")
    parser.add_argument("--sub_col", default="changed_sentence")

    parser.add_argument("--model", default="allenai/OLMo-1B-hf")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--device_map_auto", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=None)

    parser.add_argument("--old_csv")
    parser.add_argument("--new_csv")
    parser.add_argument("--compare_out", default=None)

    args = parser.parse_args()

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