## BigCodeBench rows where the model prefers the substitute

| Example           |        Original → Substitute | Delta mean NLL | Possible reason                                                                                                                            |
| ----------------- | ---------------------------: | -------------: | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `BigCodeBench/35` |               `column → col` |       `-1.172` | In pandas-style code, `for col in df.columns:` is highly idiomatic. The abbreviation `col` may be more common than the full word `column`. |
| `BigCodeBench/36` |               `column → col` |       `-0.844` | Same as above: `df.columns` strongly cues the common loop variable name `col`.                                                             |
| `BigCodeBench/16` | `log_files → matching_files` |       `-0.775` | The variable is produced by `glob.glob(...)`, so `matching_files` may better describe files matching a pattern.                            |
| `BigCodeBench/22` |          `combined → merged` |       `-0.664` | The code combines two lists using `zip_longest`; `merged` may be a more natural result name for this operation.                            |
| `BigCodeBench/23` |          `combined → merged` |       `-0.633` | Same pattern as above: two sequences are being interleaved/merged, so `merged` is semantically natural.                                    |
| `BigCodeBench/19` |         `files → file_paths` |       `-0.630` | `glob.glob(...)` returns paths, not file objects, so `file_paths` is more precise than `files`.                                            |

## QuixBugs rows where the model prefers the substitute

| Example                 |               Original → Substitute | Delta mean NLL | Possible reason                                                                                             |
| ----------------------- | ----------------------------------: | -------------: | ----------------------------------------------------------------------------------------------------------- |
| `depth_first_search`    |      `nodesvisited → visited_nodes` |       `-3.672` | `nodesvisited` is awkward and non-Pythonic; `visited_nodes` is the conventional form.                       |
| `reverse_linked_list`   |                   `prevnode → prev` |       `-2.195` | Linked-list code commonly uses `prev`, `curr`, and `next`; `prevnode` is less idiomatic.                    |
| `kth`                   |               `pivot → pivot_value` |       `-2.033` | The variable stores the pivot value, so `pivot_value` may be more explicit than `pivot`.                    |
| `lis`                   |             `longest → best_length` |       `-1.946` | `longest` is vague; `best_length` better describes the scalar being tracked.                                |
| `possible_change`       |                      `first → coin` |       `-1.578` | The code destructures a list of coins; `coin` is more semantically meaningful than generic `first`.         |
| `wrap`                  |                   `end → break_pos` |       `-1.384` | The variable comes from `text.rfind(...)`, so it marks a line-break position. `break_pos` is more specific. |
| `breadth_first_search`  |                  `queue → frontier` |       `-1.336` | In graph search, `frontier` is a common conceptual name for the set/queue of nodes to explore.              |
| `hanoi`                 |                     `steps → moves` |       `-1.055` | Tower of Hanoi outputs are conventionally called `moves`, not `steps`.                                      |
| `minimum_spanning_tree` | `group_by_node → component_by_node` |       `-0.742` | MST/union-find logic is about connected components; `component_by_node` is more algorithmically precise.    |
| `to_base`               |               `result → result_str` |       `-0.662` | The variable is initialized as an empty string, so `result_str` gives useful type information.              |
| `pascal`                |                       `r → row_idx` |       `-0.524` | In Pascal triangle generation, the loop variable indexes rows; `row_idx` is more descriptive than `r`.      |
| `shunting_yard`         |               `opstack → operators` |       `-0.334` | `opstack` is compressed and awkward; `operators` is clearer, though it loses the stack nuance.              |
| `next_palindrome`       |              `high_mid → right_mid` |       `-0.297` | `right_mid` is more naturally paired with `left_mid`; `high_mid` is less idiomatic.                         |
