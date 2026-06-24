# Cases Where Variable Renaming Prefers the Substitute

Negative delta means the substitute identifier has lower mean NLL than the original identifier, so the model prefers the substitute name in that context. These cases suggest that variable-renaming scores are strongly affected by identifier naturalness and local naming conventions, rather than only by benchmark exposure.

## BigCodeBench rows where the model prefers the substitute

| Example           |        Original → Substitute | Delta mean NLL | Likely explanation                                                                                                                                         |
| ----------------- | ---------------------------: | -------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `BigCodeBench/35` |               `column → col` |       `-1.172` | In pandas-style code, `for col in df.columns:` is highly idiomatic. The abbreviation `col` is likely more locally predictable than the full word `column`. |
| `BigCodeBench/36` |               `column → col` |       `-0.844` | Same pattern as above: `df.columns` strongly cues the conventional loop variable `col`.                                                                    |
| `BigCodeBench/16` | `log_files → matching_files` |       `-0.775` | The variable is produced by `glob.glob(...)`, so `matching_files` may better describe the result of matching a filename pattern.                           |
| `BigCodeBench/22` |          `combined → merged` |       `-0.664` | The code combines two lists using `zip_longest`; `merged` may be a more natural name for the resulting sequence.                                           |
| `BigCodeBench/23` |          `combined → merged` |       `-0.633` | Same pattern as above: two sequences are being interleaved or merged, making `merged` semantically natural.                                                |
| `BigCodeBench/19` |         `files → file_paths` |       `-0.630` | `glob.glob(...)` returns path strings rather than file objects, so `file_paths` is more precise than `files`.                                              |

## QuixBugs rows where the model prefers the substitute

| Example                 |               Original → Substitute | Delta mean NLL | Likely explanation                                                                                               |
| ----------------------- | ----------------------------------: | -------------: | ---------------------------------------------------------------------------------------------------------------- |
| `depth_first_search`    |      `nodesvisited → visited_nodes` |       `-3.672` | `nodesvisited` is awkward and non-Pythonic; `visited_nodes` is the conventional snake_case form.                 |
| `reverse_linked_list`   |                   `prevnode → prev` |       `-2.195` | Linked-list code commonly uses names like `prev`, `curr`, and `next`; `prevnode` is less idiomatic.              |
| `kth`                   |               `pivot → pivot_value` |       `-2.033` | The variable stores the pivot value, so `pivot_value` may be more explicit in this context.                      |
| `lis`                   |             `longest → best_length` |       `-1.946` | `longest` is somewhat vague; `best_length` better describes the scalar being tracked.                            |
| `possible_change`       |                      `first → coin` |       `-1.578` | The code destructures a list of coins, so `coin` is more semantically meaningful than the generic name `first`.  |
| `wrap`                  |                   `end → break_pos` |       `-1.384` | The variable comes from `text.rfind(...)`, so it represents a line-break position; `break_pos` is more specific. |
| `breadth_first_search`  |                  `queue → frontier` |       `-1.336` | In graph search, `frontier` is a common conceptual name for the set or queue of nodes still to explore.          |
| `hanoi`                 |                     `steps → moves` |       `-1.055` | Tower of Hanoi outputs are conventionally called `moves`, making the substitute more natural.                    |
| `minimum_spanning_tree` | `group_by_node → component_by_node` |       `-0.742` | MST and union-find logic is about connected components; `component_by_node` is more algorithmically precise.     |
| `to_base`               |               `result → result_str` |       `-0.662` | The variable is initialized as an empty string, so `result_str` provides useful type information.                |
| `pascal`                |                       `r → row_idx` |       `-0.524` | In Pascal-triangle generation, the loop variable indexes rows; `row_idx` is more descriptive than `r`.           |
| `shunting_yard`         |               `opstack → operators` |       `-0.334` | `opstack` is compressed and awkward; `operators` is clearer, though it loses the stack nuance.                   |
| `next_palindrome`       |              `high_mid → right_mid` |       `-0.297` | `right_mid` is more naturally paired with a left-side midpoint; `high_mid` is less idiomatic.                    |

## Interpretation

These examples show why variable renaming is a weak perturbation axis for exposure detection. The model does not simply prefer the original identifier. Instead, it often prefers the substitute when the substitute is more idiomatic, more semantically precise, or better aligned with local code conventions.

This means the variable-renaming delta is heavily confounded by identifier naturalness. A positive delta can reflect exposure, but it can also reflect the fact that the original name is more conventional or more predictable in context. A negative delta often occurs when the substitute is plainly better than the original. Therefore, variable renaming does not provide a clean seen-versus-unseen signal without much stronger control over naming quality, tokenization, and semantic equivalence.
