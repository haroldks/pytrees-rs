# Command line arguments

```dtrees_rs --help
Usage: dtrees-rs [OPTIONS] --input <INPUT> <COMMAND>

Commands:
  dl85    DL8.5 Optimal search Algorithm with no depth limit and classification error as criterion.
  d2-odt  Optimal depth 2 algorithms using Error or Information as criterion
  lgdt    Less greedy decision tree approach usind misclassification or information gain tree as sliding window
  help    Print this message or the help of the given subcommand(s)

Options:
  -i, --input <INPUT>  Dataset input file path
      --print-stats    Printing Statistics and Constraints
      --print-tree     Printing Tree
  -h, --help           Print help
  -V, --version        Print version

```

Le binaire offres plusieurs sous commandes en fonction de l'algorithm de recherche utilisé.

- [dl85](#learn-an-optimal-decision-tree-using-dl85): Generate decision tree usind DL8.5.
- [d2-odt](#learning-depth-2-optimal-decision-trees): Generate only optimal decision trees with a depth of 2, using either classification error or information gain as the cost function.
- [lgdt](#learning-less-greedy-decision-trees): Generate Lookahead Trees for decision tree.


### Learn an Optimal Decision Tree using DL8.5

```dtrees_rs dl85 --help
Usage: dtrees-rs --input <INPUT> dl85 [OPTIONS] --depth <DEPTH>

Options:
  -s, --support <SUPPORT>
          Minimum support [default: 1]
  -d, --depth <DEPTH>
          Maximum depth
      --sorting-once
          Sorting Features based on heuristic only at the root (true) or at each node
      --specialization <SPECIALIZATION>
          Use Murtree Specialization Algorithm [default: none] [possible values: murtree, none]
      --lb <LOWER_BOUND_HEURISTIC>
          Lower bound heuristic strategy [default: none] [possible values: similarity, none]
  -b, --branching <BRANCHING>
          Branching type [default: none] [possible values: dynamic, none]
      --cache <CACHE_TYPE>
          [default: trie] [possible values: trie, hashmap]
      --cache-init-size <CACHE_INIT_SIZE>
          Cache init size Represents the reserved starting size of the cache [default: 0]
      --init-strategy <INIT_STRATEGY>
          Cache Initialization strategy [default: none] [possible values: dynamic-allocation, user-allocation, none]
  -h, --heuristic <HEURISTIC>
          Sorting heuristic [default: none] [possible values: information-gain, information-gain-ratio, gini-index, none]
      --max-error <MAX_ERROR>
          Tree error initial upper bound [default: inf]
  -t, --timeout <TIMEOUT>
          Maximum time allowed to the search
  -h, --help
          Print help
```

### Learning Time constrained DL8.5 (tba)

### Learning Depth-2 Optimal decision Trees

```dtrees_rs d2-odt--help
Usage: dtrees-rs --input <INPUT> d2-odt [OPTIONS]

Options:
  -s, --support <SUPPORT>      Minimum support [default: 1]
  -d, --depth <DEPTH>          Depth The depth you want. The algorithm is optimized for depth 1 and 2 and won't work for more than that [default: 2]
  -o, --objective <OBJECTIVE>  Objective to optimise. Error or Information Gain [default: error] [possible values: error, information-gain]
  -h, --help                   Print help
```

### Learning Less Greedy Decision Trees

```dtrees_rs lgdt --help
Usage: dtrees-rs --input <INPUT> lgdt [OPTIONS] --depth <DEPTH>

Options:
  -s, --support <SUPPORT>      Minimum support [default: 1]
  -d, --depth <DEPTH>          Maximum depth
  -o, --objective <OBJECTIVE>  Objective function inside [default: error] [possible values: error, information-gain]
  -h, --help                   Print help
```
