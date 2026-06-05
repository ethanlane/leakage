````markdown
## JSONL files

This project uses several JSONL files at different stages of the variable-name perturbation pipeline. Each line is one benchmark example.

### 1. Raw candidate-pool JSONL

Example files:

```text
quixbugs_var_candidates.jsonl
bigcodebench_var_candidates.jsonl
bigcodebench_var_candidates_test.jsonl
````

These files are produced by the automatic candidate-generation script.

Each row contains:

```text
example_id
source_code
candidates
code_field
prefix_field
```

The `candidates` field is a list of possible local variables that can be renamed. Each candidate includes:

```text
candidate_id
old_name
role
occurrence_count
first_start_char
first_end_char
first_line
first_column
binding_context
candidate_new_names
original_continuation
```

These files are **not manually curated yet**. They may contain generic substitute names such as `tmp`, `val`, `cur`, `res`, etc., so they are not used directly for scoring.

---

### 2. Semantic candidate-pool JSONL

Example files:

```text
quixbugs_var_candidates_semantic_llm.jsonl
bigcodebench_var_candidates_test_semantic_llm.jsonl
```

These are cleaned versions of the raw candidate pools.

The main change is that generic substitute names are replaced with more semantic-preserving alternatives.

Example:

```text
queue -> frontier / worklist / pending
count -> bit_count / counter / num_set_bits
random_string -> generated_string / rand_string / text
char_counts -> char_counter / counts_by_char / char_freqs
```

These files are the ones given to Claude/ChatGPT for final selection.

Important: these files still contain multiple possible candidates per example. No final perturbation has been chosen yet.

---

### 3. Full semantic audit JSONL

Example files:

```text
quixbugs_var_candidates_semantic.jsonl
bigcodebench_var_candidates_test_semantic.jsonl
```

These are the full audit versions of the semantic candidate pools.

They preserve extra information about the original candidate names and the cleaning process. They are useful for debugging, but the LLM selection step usually uses the smaller `*_semantic_llm.jsonl` files.

---

### 4. Dropped-candidate JSONL

Example files:

```text
quixbugs_var_candidates_semantic_dropped.jsonl
bigcodebench_var_candidates_test_semantic_dropped.jsonl
```

These files record candidates that were removed during semantic cleanup.

Typical dropped cases include:

```text
one-occurrence bindings
parameter reassignments
exception variables, e.g. except Exception as e
context-manager aliases, e.g. with open(...) as f
unsafe or ambiguous first-binding positions
```

These rows are not used for LLM selection or model scoring.

---

### 5. LLM-selected JSONL

Example files:

```text
quixbugs_var_selected.jsonl
bigcodebench_var_selected.jsonl
```

These files are produced by Claude/ChatGPT after reading the semantic candidate pool.

Each row contains exactly one decision:

```text
ACCEPT
```

or

```text
REJECT
```

For accepted examples, the row contains the selected rename:

```text
selected_candidate_id
old_name
new_name
first_start_char
first_end_char
first_line
first_column
binding_context
original_continuation
substitute_continuation
selection_reason
```

The `source_code` column is preserved unchanged.

Example:

```json
{
  "example_id": "bitcount",
  "decision": "ACCEPT",
  "old_name": "count",
  "new_name": "bit_count",
  "first_start_char": 21,
  "first_end_char": 26,
  "binding_context": "count = 0"
}
```

This file is the input to the model-scoring script.

---

### 6. Model-score JSONL

Example files:

```text
quixbugs_var_scores.jsonl
bigcodebench_var_scores.jsonl
```

These files are produced by the analysis script after running the target model.

For each accepted row, the script:

1. checks the selected variable position,
2. applies the scope-aware rename,
3. optionally runs tests,
4. scores only the first variable-name occurrence.

The main score fields are:

```text
orig_mean_nll
sub_mean_nll
delta_mean_nll
prefers_original
```

The metric is:

```text
delta_mean_nll = mean NLL(substitute variable name | prefix)
               - mean NLL(original variable name | prefix)
```

Interpretation:

```text
delta_mean_nll > 0
```

means the model assigns lower average NLL to the original variable name, so the model prefers the original benchmark wording.

```text
delta_mean_nll < 0
```

means the model prefers the substitute variable name.

The script also stores tokenization details:

```text
orig_num_tokens
sub_num_tokens
orig_tokens
sub_tokens
orig_token_nlls
sub_token_nlls
```

These are useful for checking whether the result is affected by tokenization length.

Important: raw NLL values such as `orig_mean_nll` and `sub_mean_nll` should never be negative. Negative `delta_mean_nll` is valid, but negative raw NLL indicates a scoring bug.

---

### 7. Summary JSON

Example files:

```text
quixbugs_var_summary.json
bigcodebench_var_summary.json
```

These files summarize the model-score JSONL.

Main fields:

```text
n_rows
n_scored
n_accept
n_reject
n_position_mismatch
n_rename_failed
n_scoring_failed
n_original_test_failed
n_perturbed_test_failed
overall
by_dataset
```

The most important result block is:

```json
"overall": {
  "n": 37,
  "mean_delta_mean_nll": ...,
  "median_delta_mean_nll": ...,
  "std_delta_mean_nll": ...,
  "frac_positive_delta_mean_nll": ...
}
```

The primary metric is:

```text
mean_delta_mean_nll
```

and the robustness check is:

```text
frac_positive_delta_mean_nll
```

---

## Recommended pipeline

The intended file flow is:

```text
raw candidate pool
  -> semantic candidate pool
  -> LLM-selected JSONL
  -> model-score JSONL
  -> summary JSON / plots
```

Concretely:

```text
quixbugs_var_candidates.jsonl
  -> quixbugs_var_candidates_semantic_llm.jsonl
  -> quixbugs_var_selected.jsonl
  -> quixbugs_var_scores.jsonl
  -> quixbugs_var_summary.json
```

and:

```text
bigcodebench_var_candidates.jsonl
  -> bigcodebench_var_candidates_semantic_llm.jsonl
  -> bigcodebench_var_selected.jsonl
  -> bigcodebench_var_scores.jsonl
  -> bigcodebench_var_summary.json
```

---

## Dataset roles

In the current experiment:

```text
QuixBugs
```

is treated as the likely-seen / leaked positive-control dataset.

```text
BigCodeBench
```

is treated as the likely-unseen control dataset.

The hypothesis is that a model exposed to a benchmark should show stronger preference for the exact original variable name:

```text
QuixBugs delta_mean_nll > BigCodeBench delta_mean_nll
```

where positive `delta_mean_nll` means the model prefers the original variable name over the substitute.

```
```
