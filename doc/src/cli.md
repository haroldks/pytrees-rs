# Command line tools

Two binaries give access to the algorithms without Python. Build them with:

```bash
cargo build --release -p dtrees-cli -p contree-cli
```

Both read datasets as text files with one row per line, values separated by
whitespace, and the class label (an integer from 0) in the first column.
Lines starting with `#` are ignored.

```text
1 1 0 0 1 0 1
0 0 1 0 1 1 0
```

## dtrees

`dtrees` runs DL8.5, LGDT and the depth-2 solver on binary features (every
value after the label must be 0 or 1). The common options come before the
subcommand:

```bash
dtrees --input data.txt --print-tree --print-stats dl85 --depth 3 --support 5 --timeout 60
dtrees --input data.txt --print-tree lgdt --depth 6
dtrees --input data.txt --print-tree d2 --depth 2
```

| Option | Description |
|---|---|
| `-i, --input` | The dataset file. |
| `--print-tree`, `--print-stats` | Print the tree and the search statistics. |

### `dl85`

| Option | Default | Description |
|---|---|---|
| `-d, --depth` | required | Maximum depth of the tree. |
| `-s, --support` | `1` | Minimum number of rows in each leaf. |
| `-t, --timeout` | none | Time limit in seconds. |
| `--heuristic` | `no-heuristic` | Feature order: `no-heuristic`, `gini-index`, `information-gain`, `weighted-entropy`. |
| `--cache` | `trie` | How solved subproblems are stored: `trie` (by path) or `cover` (by depth and rows); see [Choosing the cache](estimators/dl85.md#choosing-the-cache). |
| `--depth2-policy` | `enabled` | Use the depth-2 solver for the last two levels. |
| `--lb` | `disabled` | `similarity` enables the similarity lower bound. |
| `-b, --branching-policy` | `default` | `dynamic` searches first the branch with the higher lower bound. |
| `--always-sort` | | Sort the features by the heuristic at every node, not only at the root. |
| `--max-error` | `inf` | Initial upper bound on the error. |
| `--print-config` | | Print the configuration. |

### `lgdt` and `d2`

| Option | Default | Description |
|---|---|---|
| `-d, --depth` | required for `lgdt`, `2` for `d2` | Maximum depth (1 or 2 for `d2`). |
| `-s, --support` | `1` | Minimum number of rows in each leaf. |
| `-o, --objective` | `error` | What the tree (`d2`) or the lookahead (`lgdt`) optimises: `error` (misclassifications) or `information-gain`. |

`dtrees <command> --help` lists every option.

## con-tree

`con-tree` runs ConTree on continuous features:

```bash
con-tree --input data.txt --depth 3 --sort-by-heuristic --print-tree --print-stats
con-tree --input data.txt --depth 5 --use-lds --sort-by-heuristic --time-limit 60 --print-stats
```

| Option | Default | Description |
|---|---|---|
| `-i, --input` | required | The dataset file. |
| `-d, --depth` | required | Maximum depth of the tree. |
| `-s, --support` | `1` | Minimum number of rows in each leaf. |
| `-t, --time-limit` | `600` | Time limit in seconds. |
| `--max-gap` | `0` | Error gap to the optimum that is tolerated. |
| `--max-error` | none | Initial upper bound on the error. |
| `--sort-by-heuristic` | | Explore features and thresholds in Gini order. |
| `--split-selection-strategy` | `mid` | `mid`, `first` or `random`; see [ConTreeClassifier](estimators/contree.md). |
| `--no-fast-d2` | | Disable the depth-2 solver. |
| `--use-lds` | | Use the anytime search. |
| `--budget-schedule` | `diagonal` with `first`, `square` otherwise | `diagonal` widens the feature discrepancy and the number of thresholds tried together; `square` widens only the discrepancy when the selector is `mid` or `random`, which use no threshold budget after the first pass. `con-tree` warns about `diagonal` with those selectors: most of its passes would repeat an earlier one. |
| `--lookahead` | none | SPLIT: search only the top levels and complete each node below them with a greedy tree (lowest Gini first). Runs the anytime search, with its budgets only if `--use-lds` is given. The depth-2 solver still solves any node with two levels left, so a lookahead of depth − 2 or more is the exact search. |
| `--anytime` | | With `--lookahead`: one level deeper between passes, until the whole depth is searched and the tree is optimal. |
| `--sync` | | With `--anytime --use-lds`: spread the deepenings over the budget schedule, instead of deepening after every pass. |
| `--no-lookahead-pruning` | | With `--lookahead`: run the interval pruner only in passes that search the whole depth. Its rules assume optimal subtrees, so in other passes it may skip a threshold whose greedy completion is better. |
| `--print-tree`, `--print-stats` | | Print the tree, and the statistics with the reason the search stopped. |

The four anytime modes:

```bash
con-tree --input data.txt --depth 5 --use-lds                              # LDS
con-tree --input data.txt --depth 5 --lookahead 1                          # SPLIT, one pass
con-tree --input data.txt --depth 5 --lookahead 0 --anytime                # anytime SPLIT
con-tree --input data.txt --depth 5 --lookahead 0 --anytime --use-lds      # both
```

Which one to use: on 16 datasets with continuous features, at depths 4 and
5 with a 60 s limit, the anytime SPLIT modes had the best trees early on: the
average primal gap at 5 s was 22–30 % for anytime SPLIT, alone or with LDS,
against 32–41 % for LDS alone (the primal gap of a run at a time is how far
its best tree's error is from the best error any method reached, as a share
of the larger of the two, averaged over the time so far). By 60 s all of them
were within a point at depth 4, and at depth 5 LDS and anytime SPLIT with LDS
(21.8 %) were ahead of anytime SPLIT alone (23.2 %). All with the `mid`
selector and Gini ordering.

A pass that completes nodes greedily never proves optimality: plain SPLIT
ends with status `budget exhausted` unless its tree has no error.
