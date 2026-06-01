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
from datasets import load_dataset


HENDRYCKS_DATASET = "EleutherAI/hendrycks_math"

# Default mapping for your unseen math_full.jsonl dataset labels.
# You can override/add mappings with --hf_map_json if needed.
DEFAULT_HF_MAP = {
    "smt_2025": {"hf_name": "MathArena/smt_2025", "split": "train"},
    "aime_2025": {"hf_name": "MathArena/aime_2025", "split": "train"},
    "hmmt_feb_2025": {"hf_name": "MathArena/hmmt_feb_2025", "split": "train"},
    "cmimc_2025": {"hf_name": "MathArena/cmimc_2025", "split": "train"},

    # SMT 2024 is not clearly in MathArena under this exact name.
    # This public HF dataset appears to use fields like id/category/question/answer.
    "smt_2024": {"hf_name": "nmayorga7/smt-2024", "split": "train"},
}

_DATASET_CACHE = {}


def safe_str(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def first_existing_key(row, keys):
    for k in keys:
        if k in row and row[k] is not None and safe_str(row[k]) != "":
            return k
    return None


def load_hf_map(path):
    mapping = dict(DEFAULT_HF_MAP)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user_map = json.load(f)
        mapping.update(user_map)
    return mapping


def get_hf_dataset(hf_name, split="train", config=None):
    key = (hf_name, config, split)
    if key not in _DATASET_CACHE:
        if config:
            _DATASET_CACHE[key] = load_dataset(hf_name, config, split=split)
        else:
            _DATASET_CACHE[key] = load_dataset(hf_name, split=split)
    return _DATASET_CACHE[key]


def get_problem_text_from_example(ex):
    for k in ["problem", "question", "text", "prompt"]:
        if k in ex and safe_str(ex[k]):
            return safe_str(ex[k]), k
    return "", None


def get_solution_from_example(ex):
    for k in ["solution", "answer", "final_answer", "gold_answer"]:
        if k in ex and safe_str(ex[k]):
            return ex[k]
    return ""


def find_hf_example_by_source_id(ds, source_id, original_number=None):
    """
    Tries to find the correct HF row from source_id.

    Handles:
      - MathArena problem_idx
      - datasets with id
      - direct row index
      - one-based row index fallback
    """
    sid = safe_str(source_id)

    # 1. Exact match against common id fields.
    id_fields = ["problem_idx", "id", "source_id", "idx"]

    for i, ex in enumerate(ds):
        for field in id_fields:
            if field in ex and safe_str(ex[field]) == sid:
                return i, ex, f"{field}_exact"

    # 2. Integer source_id fallback: row index, then one-based index.
    if re.fullmatch(r"\d+", sid):
        n = int(sid)

        if 0 <= n < len(ds):
            return n, ds[n], "row_index_0_based"

        if 1 <= n <= len(ds):
            return n - 1, ds[n - 1], "row_index_1_based"

    # 3. Last resort: if original_number appears uniquely-ish, search candidate rows.
    # This is weaker, but can rescue files whose source_id does not match HF indexing.
    if original_number:
        candidates = []
        for i, ex in enumerate(ds):
            problem, _ = get_problem_text_from_example(ex)
            if original_number in problem:
                candidates.append((i, ex))

        if len(candidates) == 1:
            return candidates[0][0], candidates[0][1], "unique_original_number_match"

    return None, None, "not_found"


def hydrate_hendrycks(row):
    """
    source_id example:
      algebra/test/36
    """
    source_id = safe_str(row.get("source_id"))
    parts = source_id.split("/")

    if len(parts) != 3:
        return row, "bad_hendrycks_source_id"

    config, split, idx_str = parts
    idx = int(idx_str)

    ds = get_hf_dataset(HENDRYCKS_DATASET, split=split, config=config)

    if idx < 0 or idx >= len(ds):
        return row, "hendrycks_index_out_of_range"

    ex = ds[idx]
    problem, problem_field = get_problem_text_from_example(ex)

    if not problem:
        return row, "hendrycks_problem_missing"

    out = dict(row)
    out["problem"] = problem
    out["solution"] = get_solution_from_example(ex)
    out["hf_name"] = HENDRYCKS_DATASET
    out["hf_config"] = config
    out["hf_split"] = split
    out["hf_row_index"] = idx
    out["hf_problem_field"] = problem_field
    out["hydration_method"] = "hendrycks_source_id"

    return out, None


def hydrate_generic_hf(row, hf_map):
    """
    Handles math_full rows like:
      {"dataset":"smt_2025","source_id":"44", ...}
    """
    dataset_label = safe_str(row.get("dataset"))
    source_id = safe_str(row.get("source_id"))
    original_number = safe_str(row.get("original_number"))

    if dataset_label not in hf_map:
        return row, f"no_hf_map_for_dataset={dataset_label}"

    spec = hf_map[dataset_label]
    hf_name = spec["hf_name"]
    split = spec.get("split", "train")
    config = spec.get("config", None)

    ds = get_hf_dataset(hf_name, split=split, config=config)

    idx, ex, method = find_hf_example_by_source_id(
        ds,
        source_id=source_id,
        original_number=original_number,
    )

    if ex is None:
        return row, f"hf_source_id_not_found:{dataset_label}/{source_id}"

    problem, problem_field = get_problem_text_from_example(ex)

    if not problem:
        return row, f"hf_problem_missing:{dataset_label}/{source_id}"

    out = dict(row)
    out["problem"] = problem
    out["solution"] = get_solution_from_example(ex)
    out["hf_name"] = hf_name
    out["hf_config"] = config
    out["hf_split"] = split
    out["hf_row_index"] = idx
    out["hf_match_method"] = method
    out["hf_problem_field"] = problem_field
    out["hydration_method"] = "generic_hf"

    return out, None


def hydrate_row(row, hf_map):
    # Already has problem text.
    if any(k in row and safe_str(row[k]) for k in ["problem", "question", "text", "prompt"]):
        return row, None

    dataset_label = safe_str(row.get("dataset"))
    source_id = safe_str(row.get("source_id"))

    if dataset_label == "hendrycks_math" or len(source_id.split("/")) == 3:
        return hydrate_hendrycks(row)

    return hydrate_generic_hf(row, hf_map)


def locate_number_span(row, text, original_number):
    """
    Prefer explicit offsets if available. Otherwise require unique occurrence.
    """
    offset_keys = [
        "changed_start",
        "original_number_offset",
        "number_offset",
        "target_offset",
        "start",
        "span_start",
        "original_start",
    ]

    end_keys = [
        "changed_end",
        "original_number_end",
        "number_end",
        "target_end",
        "end",
        "span_end",
        "original_end",
    ]

    off_key = first_existing_key(row, offset_keys)
    end_key = first_existing_key(row, end_keys)

    if off_key is not None:
        start = int(row[off_key])
        end = int(row[end_key]) if end_key is not None else start + len(original_number)

        if not (0 <= start < end <= len(text)):
            return None, None, "offset_out_of_bounds"

        surface = text[start:end]

        if surface == original_number or surface.strip() == original_number.strip():
            return start, end, None

        # If the HF text formatting differs from the file used to create offsets,
        # fall back to unique occurrence.
        matches = list(re.finditer(re.escape(original_number), text))
        if len(matches) == 1:
            m = matches[0]
            return m.start(), m.end(), "offset_mismatch_but_unique_fallback"

        return None, None, f"offset_mismatch_surface={surface!r}"

    matches = list(re.finditer(re.escape(original_number), text))

    if len(matches) == 0:
        return None, None, "original_number_not_found"

    if len(matches) > 1:
        return None, None, "original_number_multiple_occurrences"

    m = matches[0]
    return m.start(), m.end(), None


def replace_span(text, start, end, replacement):
    return text[:start] + replacement + text[end:]


def load_model_and_tokenizer(model_name, dtype="bf16", trust_remote_code=False, device_map_auto=False):
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


def span_logprob_from_forward(fwd, text, start, end):
    input_ids = fwd["input_ids"]
    offsets = fwd["offsets"]
    log_probs = fwd["log_probs"]

    indices = []

    for i, (a, b) in enumerate(offsets):
        if b <= start or a >= end:
            continue

        if a < start and text[a:start].strip():
            return None, None, None, "token_crosses_left_nonspace"

        if b > end and text[end:b].strip():
            return None, None, None, "token_crosses_right_nonspace"

        indices.append(i)

    if not indices:
        return None, None, None, "no_span_tokens"

    ids = [input_ids[i] for i in indices]

    total_logp = 0.0

    for i in indices:
        if i == 0:
            return None, ids, None, "no_context_for_first_token"

        tok_id = input_ids[i]
        total_logp += log_probs[0, i - 1, tok_id].item()

    return total_logp, ids, indices, None


def mean_nll_from_forward(fwd):
    input_ids = fwd["input_ids"]
    log_probs = fwd["log_probs"]

    total_nll = 0.0
    count = 0

    for i in range(1, len(input_ids)):
        tok_id = input_ids[i]
        total_nll += -log_probs[0, i - 1, tok_id].item()
        count += 1

    return total_nll / max(count, 1), total_nll, count


def summarize(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return {"n": 0}

    pos = vals[vals > 0]

    return {
        "n": int(len(vals)),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "min": float(vals.min()),
        "max": float(vals.max()),
        "frac_positive": float((vals > 0).mean()),
        "positive_half_mean": float(pos.mean()) if len(pos) else None,
    }


def run(args):
    hf_map = load_hf_map(args.hf_map_json)

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        device_map_auto=args.device_map_auto,
    )

    rows = load_jsonl(args.input)
    out_rows = []
    skip = Counter()

    for idx, row in tqdm(list(enumerate(rows)), desc=args.dataset_label):
        row, hydrate_err = hydrate_row(row, hf_map)
        if hydrate_err is not None:
            skip[hydrate_err] += 1
            continue

        text_key = first_existing_key(row, [args.text_col, "problem", "question", "text", "prompt"])
        if text_key is None:
            skip["missing_problem_text"] += 1
            continue

        orig_num_key = first_existing_key(row, [args.orig_num_col, "original_number", "orig_number"])
        pert_num_key = first_existing_key(row, [args.pert_num_col, "perturbed_number", "new_number"])

        if orig_num_key is None or pert_num_key is None:
            skip["missing_number_fields"] += 1
            continue

        original_problem = safe_str(row[text_key])
        original_number = safe_str(row[orig_num_key]).strip()
        perturbed_number = safe_str(row[pert_num_key]).strip()

        if not original_problem or not original_number or not perturbed_number:
            skip["empty_fields"] += 1
            continue

        start, end, locate_note = locate_number_span(row, original_problem, original_number)

        if start is None:
            skip[locate_note] += 1
            continue

        perturbed_problem = replace_span(original_problem, start, end, perturbed_number)
        pert_start = start
        pert_end = start + len(perturbed_number)

        orig_fwd, err = forward_text(model, tokenizer, original_problem)
        if err is not None:
            skip[f"orig_{err}"] += 1
            continue

        pert_fwd, err = forward_text(model, tokenizer, perturbed_problem)
        if err is not None:
            skip[f"pert_{err}"] += 1
            continue

        orig_logp, orig_ids, _, err = span_logprob_from_forward(
            orig_fwd,
            original_problem,
            start,
            end,
        )
        if err is not None:
            skip[f"orig_num_{err}"] += 1
            continue

        pert_logp, pert_ids, _, err = span_logprob_from_forward(
            pert_fwd,
            perturbed_problem,
            pert_start,
            pert_end,
        )
        if err is not None:
            skip[f"pert_num_{err}"] += 1
            continue

        orig_context_mean_nll, orig_context_sum_nll, orig_context_num_tokens = mean_nll_from_forward(orig_fwd)
        pert_context_mean_nll, pert_context_sum_nll, pert_context_num_tokens = mean_nll_from_forward(pert_fwd)

        orig_num_nll = -orig_logp
        pert_num_nll = -pert_logp

        delta_sum_nll = pert_num_nll - orig_num_nll
        delta_avg_nll = (pert_num_nll / len(pert_ids)) - (orig_num_nll / len(orig_ids))
        context_score = pert_context_mean_nll - orig_context_mean_nll
        count_mismatch = len(orig_ids) != len(pert_ids)

        out = dict(row)
        out.update({
            "row_idx": idx,
            "dataset_label": args.dataset_label,

            "original_problem": original_problem,
            "perturbed_problem": perturbed_problem,

            "original_number": original_number,
            "perturbed_number": perturbed_number,

            "original_number_offset": start,
            "original_number_end": end,
            "perturbed_number_offset": pert_start,
            "perturbed_number_end": pert_end,
            "locate_note": locate_note,

            "orig_num_logp": orig_logp,
            "pert_num_logp": pert_logp,
            "orig_num_nll": orig_num_nll,
            "pert_num_nll": pert_num_nll,

            "orig_num_tokens": len(orig_ids),
            "pert_num_tokens": len(pert_ids),
            "orig_num_token_ids": json.dumps(orig_ids),
            "pert_num_token_ids": json.dumps(pert_ids),
            "orig_num_token_strs": json.dumps(tokenizer.convert_ids_to_tokens(orig_ids), ensure_ascii=False),
            "pert_num_token_strs": json.dumps(tokenizer.convert_ids_to_tokens(pert_ids), ensure_ascii=False),

            "count_mismatch": count_mismatch,

            "delta_sum_nll": delta_sum_nll,
            "delta_avg_nll": delta_avg_nll,

            "orig_context_mean_nll": orig_context_mean_nll,
            "pert_context_mean_nll": pert_context_mean_nll,
            "orig_context_avg_logp": -orig_context_mean_nll,
            "pert_context_avg_logp": -pert_context_mean_nll,
            "context_score": context_score,

            "orig_context_sum_nll": orig_context_sum_nll,
            "pert_context_sum_nll": pert_context_sum_nll,
            "orig_context_num_tokens": orig_context_num_tokens,
            "pert_context_num_tokens": pert_context_num_tokens,
        })

        out_rows.append(out)

        if args.max_examples is not None and len(out_rows) >= args.max_examples:
            break

    write_jsonl(out_rows, args.out)

    all_scores = [r["delta_sum_nll"] for r in out_rows]
    clean_scores = [
        r["delta_sum_nll"]
        for r in out_rows
        if not r["count_mismatch"]
        and r["orig_num_tokens"] == r["pert_num_tokens"]
    ]
    avg_scores = [r["delta_avg_nll"] for r in out_rows]
    context_scores = [r["context_score"] for r in out_rows]

    summary = {
        "input": args.input,
        "output": args.out,
        "dataset_label": args.dataset_label,
        "model": args.model,
        "input_rows": len(rows),
        "kept_rows": len(out_rows),
        "skip_counts": dict(skip),
        "delta_sum_nll": summarize(all_scores),
        "delta_sum_nll_clean_same_token_count": summarize(clean_scores),
        "delta_avg_nll": summarize(avg_scores),
        "context_score": summarize(context_scores),
    }

    summary_path = args.out.replace(".jsonl", ".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved results: {args.out}")
    print(f"Saved summary: {summary_path}")


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


def auc_seen_higher(seen_scores, unseen_scores):
    seen_scores = np.asarray(seen_scores, dtype=float)
    unseen_scores = np.asarray(unseen_scores, dtype=float)

    if len(seen_scores) == 0 or len(unseen_scores) == 0:
        return None

    scores = np.concatenate([seen_scores, unseen_scores])
    labels = np.concatenate([np.ones(len(seen_scores)), np.zeros(len(unseen_scores))])

    ranks = average_ranks(scores)
    n_pos = len(seen_scores)
    n_neg = len(unseen_scores)

    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def cohens_d(seen_scores, unseen_scores):
    seen_scores = np.asarray(seen_scores, dtype=float)
    unseen_scores = np.asarray(unseen_scores, dtype=float)

    if len(seen_scores) < 2 or len(unseen_scores) < 2:
        return None

    n1, n2 = len(seen_scores), len(unseen_scores)
    s1 = seen_scores.std(ddof=1)
    s2 = unseen_scores.std(ddof=1)

    pooled = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled == 0:
        return None

    return float((seen_scores.mean() - unseen_scores.mean()) / pooled)


def get_scores(rows, key, clean=False):
    vals = []

    for r in rows:
        if clean:
            if r.get("count_mismatch", False):
                continue
            if r.get("orig_num_tokens") != r.get("pert_num_tokens"):
                continue

        v = r.get(key)
        if v is not None:
            try:
                v = float(v)
                if math.isfinite(v):
                    vals.append(v)
            except Exception:
                pass

    return np.asarray(vals, dtype=float)


def compare(args):
    seen_rows = load_jsonl(args.seen_jsonl)
    unseen_rows = load_jsonl(args.unseen_jsonl)

    seen_scores = get_scores(seen_rows, args.score_key, clean=args.clean)
    unseen_scores = get_scores(unseen_rows, args.score_key, clean=args.clean)

    out = {
        "score_key": args.score_key,
        "clean": args.clean,
        "seen_file": args.seen_jsonl,
        "unseen_file": args.unseen_jsonl,
        "seen": summarize(seen_scores),
        "unseen": summarize(unseen_scores),
        "difference_mean_seen_minus_unseen": (
            float(seen_scores.mean() - unseen_scores.mean())
            if len(seen_scores) and len(unseen_scores)
            else None
        ),
        "auc_seen_higher_than_unseen": auc_seen_higher(seen_scores, unseen_scores),
        "cohens_d_seen_minus_unseen": cohens_d(seen_scores, unseen_scores),
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))

    if args.compare_out:
        os.makedirs(os.path.dirname(args.compare_out) or ".", exist_ok=True)
        with open(args.compare_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Saved compare summary: {args.compare_out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["run", "compare"], default="run")

    # Run mode
    p.add_argument("--input")
    p.add_argument("--out")
    p.add_argument("--dataset_label", default="dataset")

    p.add_argument("--hf_map_json", default=None)

    p.add_argument("--model", default="allenai/OLMo-1B-hf")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--device_map_auto", action="store_true")
    p.add_argument("--max_examples", type=int, default=None)

    p.add_argument("--text_col", default="problem")
    p.add_argument("--orig_num_col", default="original_number")
    p.add_argument("--pert_num_col", default="perturbed_number")

    # Compare mode
    p.add_argument("--seen_jsonl")
    p.add_argument("--unseen_jsonl")
    p.add_argument("--score_key", default="delta_sum_nll")
    p.add_argument("--clean", action="store_true")
    p.add_argument("--compare_out")

    args = p.parse_args()

    if args.mode == "run":
        if not args.input or not args.out:
            raise ValueError("--input and --out are required in run mode")
        run(args)
    else:
        if not args.seen_jsonl or not args.unseen_jsonl:
            raise ValueError("--seen_jsonl and --unseen_jsonl are required in compare mode")
        compare(args)


if __name__ == "__main__":
    main()