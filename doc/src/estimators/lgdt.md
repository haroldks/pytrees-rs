# LGDTClassifier

`LGDTClassifier` grows a tree top-down, like CART, but chooses each test with
a two-level lookahead. At each node it computes the best tree of depth 2 for
the rows in the node, keeps only its root test, and repeats on each child.
A single greedy split can miss tests that only pay off one level down (XOR is
the classic example); a depth-2 lookahead does not.

The lookahead uses the same specialised depth-2 solver as DL8.5, which counts
the classes of every pair of features once and derives every depth-2 tree from
those counts. That keeps LGDT fast enough for deep trees on large datasets,
where an optimal search would not finish. Features must be binary.

```python
from pytrees import LGDTClassifier

clf = LGDTClassifier(max_depth=8, min_sup=5).fit(X, y)
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `min_sup` | `1` | Minimum number of training rows in each leaf. |
| `max_depth` | `2` | Maximum depth of the tree. At depth 2 or less the tree is optimal (for `criterion="error"`). |
| `criterion` | `"error"` | What the depth-2 lookahead optimises: `"error"` (misclassifications) or `"information_gain"`. |

## Fitted attributes

`classes_`, `n_classes_`, `n_features_in_`, `tree_` and `train_error_`, as
described for [DL85Classifier](dl85.md). `statistics_` holds the error,
duration and sizes of the problem.

## Reference

H. Kiossou, P. Schaus, S. Nijssen and G. Aglin. *Efficient Lookahead Decision
Trees.* IDA 2024.
