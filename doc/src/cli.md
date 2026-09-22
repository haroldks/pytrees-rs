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
| `--budget-schedule` | `diagonal` | `diagonal` or `square`. |
| `--print-tree`, `--print-stats` | | Print the tree, and the statistics with the reason the search stopped. |
