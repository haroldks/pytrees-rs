# pytrees-rs

pytrees-rs learns anytime optimal decision trees. It is written mostly in
Rust and comes with a Python wrapper that follows the scikit-learn API.

An optimal tree is the one with the lowest training error among all trees of
a given depth. Finding it can take a long time, so the optimal searches can run
as anytime searches: they return a good tree quickly and keep improving it
until it is proven optimal or the time limit is reached. `status_` tells you
which.

| Estimator | Features | What it learns |
|---|---|---|
| `DL85Classifier` | binary | Optimal trees (DL8.5), for the misclassification error or your own |
| `ConTreeClassifier` | continuous | Optimal trees without binarising the data (ConTree) |
| `LGDTClassifier` | binary | Trees grown top-down, each test chosen with a depth-2 lookahead (LGDT) |
| `DL85Cluster` | binary | Clusterings whose clusters are the leaves of an optimal tree |

The estimators behave like any scikit-learn estimator: they can be cloned,
pickled and used in `Pipeline`, `GridSearchCV` or `cross_val_score`. The
fitted tree is in `tree_`, with scikit-learn's layout, and a row goes left
when `x[feature] <= threshold`.

## Installation

```bash
pip install pytrees-rs
```

Wheels are provided for Linux, macOS and Windows, for Python 3.10 and later.
To build from source you need a Rust toolchain (1.77 or later):

```bash
git clone https://github.com/haroldks/pytrees-rs.git
cd pytrees-rs
pip install .
```

## Quick start

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from pytrees import ConTreeClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = ConTreeClassifier(max_depth=3, min_sup=5).fit(X_train, y_train)
print(clf.status_)              # "optimal", or "time_limit" if it ran out of time
print(clf.train_error_)         # training misclassifications
print(clf.score(X_test, y_test))
print(clf.to_dot())             # the tree in Graphviz format
```

DL8.5 and LGDT need binary features (0 or 1). A `Binarizer` or
`KBinsDiscretizer` with one-hot output in a `Pipeline` takes care of that:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer
from pytrees import DL85Classifier

model = make_pipeline(
    KBinsDiscretizer(n_bins=4, encode="onehot-dense"),
    DL85Classifier(max_depth=3, min_sup=5, max_time=60),
)
model.fit(X_train, y_train)
```

### Anytime search

`fit_anytime` works like `fit`, but calls you back each time the search finds
a better tree:

```python
from pytrees import DL85Classifier
from pytrees.rules import DiscrepancyRule

X_bin = KBinsDiscretizer(n_bins=4, encode="onehot-dense").fit_transform(X)
clf = DL85Classifier(
    max_depth=5,
    heuristic="information_gain",
    discrepancy=DiscrepancyRule(),   # limited discrepancy search
    max_time=120,
)
clf.fit_anytime(X_bin, y, callback=lambda error, seconds, status: print(seconds, error, status))
```

`ConTreeClassifier` has the same method, and `use_lds=True` makes its plain
`fit` anytime too.

### Custom error functions

DL8.5 finds the tree that minimises the sum of its leaf errors, and the error
of a leaf does not have to be the number of misclassified rows. Pass your own
as `error_function`: it receives the class counts of a leaf (or its row
indices, with `error_function_input="indices"`) and returns the error and the
predicted class. Class-dependent costs, sample weights and clustering
objectives (`DL85Cluster` works this way) all fit:

```python
import numpy as np

y_bin = (y == 2).astype(int)   # is it Iris virginica?
costs = np.array([1.0, 5.0])   # missing a virginica costs five times more

def cost_sensitive(class_counts):
    counts = np.asarray(class_counts, dtype=float)
    per_prediction = [(costs * counts).sum() - costs[k] * counts[k] for k in range(len(counts))]
    best = int(np.argmin(per_prediction))
    return per_prediction[best], best

clf = DL85Classifier(max_depth=3, error_function=cost_sensitive).fit(X_bin, y_bin)
```

## Documentation

- [User guide](https://haroldks.github.io/pytrees-rs/): installation, every
  estimator and its parameters, the anytime search rules, custom error
  functions and the command line tools.
- [Rust API](https://haroldks.github.io/pytrees-rs/api/): the `dtrees-rs` and
  `contree-rs` crates.

## Repository layout

The Python package is built from a Cargo workspace:

| Path | Contents |
|---|---|
| `crates/dtrees` | The `dtrees-rs` library: DL8.5, LGDT and the search rules, over binary features |
| `crates/contree` | The `contree-rs` library: ConTree and its anytime variant, over continuous features |
| `crates/dtrees-cli`, `crates/contree-cli` | Command line front ends |
| `crates/pytrees-py` | The Python bindings (`pytrees._native`) |
| `python/pytrees` | The Python package and its scikit-learn estimators |
| `doc` | The documentation site (mdBook) |

To work on it:

```bash
cargo test --workspace                     # Rust tests
pip install maturin && maturin develop     # build the Python package in place
pytest python/tests                        # Python tests
```

## Publications

The algorithms in this repository come from the following papers. If you use
them in your work, please cite the relevant one.

- H. Kiossou, P. Schaus, S. Nijssen and V. R. Houndji.
  **Time Constrained DL8.5 Using Limited Discrepancy Search.**
  ECML PKDD 2022, LNCS 13717, pp. 443-459.
  [doi:10.1007/978-3-031-26419-1_27](https://doi.org/10.1007/978-3-031-26419-1_27)
  (`DL85Classifier` with `DiscrepancyRule`)
- H. Kiossou, P. Schaus, S. Nijssen and G. Aglin.
  **Efficient Lookahead Decision Trees.**
  IDA 2024, pp. 133-144.
  [doi:10.1007/978-3-031-58553-1_11](https://doi.org/10.1007/978-3-031-58553-1_11)
  (`LGDTClassifier`)
- H. Kiossou and P. Schaus.
  **A Generic Complete Anytime Beam Search for Optimal Decision Tree.**
  IDA 2026.
  [doi:10.1007/978-3-032-23833-7_8](https://doi.org/10.1007/978-3-032-23833-7_8),
  [arXiv:2508.06064](https://arxiv.org/abs/2508.06064)
  (the search rules of `DL85Classifier`: CA-DL8.5)
- H. Kiossou, P. Schaus and S. Nijssen.
  **Anytime Optimal Decision Tree Learning with Continuous Features.**
  ECML PKDD 2026.
  [arXiv:2601.14765](https://arxiv.org/abs/2601.14765)
  (`ConTreeClassifier` with `use_lds=True`)

They build on:

- G. Aglin, S. Nijssen and P. Schaus. Learning Optimal Decision Trees Using
  Caching Branch-and-Bound Search. AAAI 2020. (DL8.5; the original
  implementation is [pydl8.5](https://github.com/aia-uclouvain/pydl8.5).)
- E. Demirović, A. Lukina, E. Hebrard, J. Chan, J. Bailey, C. Leckie,
  K. Ramamohanarao and P. J. Stuckey. MurTree: Optimal Decision Trees via
  Dynamic Programming and Search. JMLR 23, 2022. (The depth-2 solver.)
- C. E. Briţa, J. G. M. van der Linden and E. Demirović. Optimal
  Classification Trees for Continuous Feature Data Using Dynamic Programming
  with Branch-and-Bound. AAAI 2025. (ConTree; the original implementation is
  [ConSol-Lab/contree](https://github.com/ConSol-Lab/contree).)

## License

MIT; see [LICENSE](LICENSE).
