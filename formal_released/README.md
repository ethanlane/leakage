# Leakage via Counterfactual Perturbations

This repository contains scripts for measuring whether a language model assigns higher probability to an original benchmark span than to a minimally perturbed counterfactual span.

The core idea is simple: perturb one localized part of an example, score only the changed span, and compare the negative log likelihood (NLL) of the original span against the perturbed span.

```text
delta_sum_nll = NLL(perturbed_span) - NLL(original_span)
```

Interpretation:

```text
delta_sum_nll > 0  -> the model assigns higher probability to the original span
delta_sum_nll < 0  -> the model assigns higher probability to the perturbed span
```

We include two experiment types:

- Math: perturb numeric values in math word problems and score only the changed number tokens.
- Code: perturb buggy code spans and score only the changed code tokens.

## Repository Structure

```text
scripts/
  run_math.py   # math number-span logprob experiment
  run_code.py   # code counterfactual-span logprob experiment
```

## Installation

The scripts use Hugging Face models and datasets.

```bash
pip install torch transformers datasets tqdm numpy
```

If you need access to gated Hugging Face models or datasets, set:

```bash
export HF_TOKEN=your_huggingface_token
```

A GPU is recommended for all nontrivial runs.

## Math Experiment

`scripts/run_math.py` compares original and perturbed numbers in paired math problems.

Expected fields for each row:

```text
original_problem  # original math problem text
problem           # perturbed math problem text
```

Optional fields used in outputs:

```text
idx
level
type
original_answer
answer
```

The script extracts numbers from both versions, aligns them by order, keeps changed number pairs, and computes NLL only on those number tokens.

### Run on the default math dataset

By default, the script uses `stellaathena/math_perturbed_200`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_math.py \
  --model Qwen/Qwen2.5-1.5B \
  --levels "Level 1,Level 2,Level 3" \
  --out_jsonl results/math_qwen15b_numlogprob.jsonl
```

### Run on a local paired JSONL file

For local JSONL files, pass the file path through `--dataset`. Use `--levels all` if the file does not contain a `level` column.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_math.py \
  --model YU-MO/Yumo-nano \
  --dataset generated/manual_natural_number_perturbed_52.jsonl \
  --levels all \
  --out_jsonl results/manual_natural_yumo_numlogprob.jsonl
```

## Code Experiment

`scripts/run_code.py` compares original buggy code spans against counterfactual buggy spans. It works with paired code counterfactual JSONL files such as QuixBugs or DebugBench-style examples.

Expected fields for each row:

```text
original_buggy_program
counterfactual_buggy_program
buggy_span              # or original_span
counterfactual_span
```

Optional fields used for more reliable span location:

```text
buggy_line
counterfactual_line
span_start
span_end
line_index
```

### Run on QuixBugs-style counterfactuals

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_code.py \
  --model bigcode/starcoderbase-3b \
  --input_jsonl generated/quixbugs_manual_counterfactual_full.jsonl \
  --out_jsonl results/quixbugs_starcoderbase3b_spanlogprob.jsonl
```

### Run on DebugBench-style counterfactuals

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_code.py \
  --model bigcode/starcoderbase-3b \
  --input_jsonl generated_debugbench/debugbench_counterfactual_curated_llm_fixed43.jsonl \
  --out_jsonl results/debugbench_starcoderbase3b_spanlogprob.jsonl
```

## Outputs

Each script writes a JSONL file with per-example or per-span scores and a summary JSON file next to it.

For example, if the output path is:

```text
results/math_qwen15b_numlogprob.jsonl
```

then the summary path defaults to:

```text
results/math_qwen15b_numlogprob.summary.json
```

Important output fields include:

```text
orig_sum_nll / orig_mean_nll
pert_sum_nll / pert_mean_nll      # math
cf_sum_nll / cf_mean_nll          # code
delta_sum_nll
delta_mean_nll
```

For both math and code experiments, positive `delta_sum_nll` means the model preferred the original span over the perturbed span.

## Notes

- These scripts measure span-level likelihood differences, not task accuracy.
- For math, number spans are aligned by order, so the paired examples should preserve the surrounding problem structure.
- For code, providing `line_index` and span offsets improves span localization when the same token appears multiple times.
- Use `--max_samples` in `run_math.py` for quick smoke tests before running full experiments.
- Use `--skip_token_count_mismatch` in `run_code.py` if you want to exclude code examples where original and counterfactual spans tokenize to different lengths.
