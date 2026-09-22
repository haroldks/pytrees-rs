#!/usr/bin/env python3
"""Check that each baseline tree reproduces the error the search reported.

`Statistics.error` is a counter maintained by the search, while the tree is
rebuilt from the cache afterwards; if they disagree, the tree handed to a user
is not the tree that was optimized.

It also pins the routing convention, `left = x <= threshold`, as in
scikit-learn. The search partitions by position (the sorted prefix
`[0, split_point)` goes left, in `view.rs`) and places each threshold
between the last value that goes left and the first that goes right
(`shared::threshold_between`), so no training value sits on a threshold.

    crates/contree/tests/baseline/check_predictions.py [baseline-dir]
"""

import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DATASETS = pathlib.Path(
    os.environ.get("CONTREE_DATASETS", HERE.parents[3] / "datasets")
)
_datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a dataset. Label is column 0; the remaining columns are features."""
    if name not in _datasets:
        arr = np.loadtxt(DATASETS / f"{name}.txt")
        _datasets[name] = (arr[:, 1:], arr[:, 0].astype(int))
    return _datasets[name]


def predict_one(node: dict, x: np.ndarray):
    while "leaf" not in node:
        node = node["left"] if x[node["feature"]] <= node["split"] else node["right"]
    return node["leaf"]


def main(argv: list[str]) -> int:
    directory = pathlib.Path(argv[1]) if len(argv) > 1 else HERE / "expected"
    records = sorted(directory.glob("*.json"))
    if not records:
        print(f"no baseline records in {directory}", file=sys.stderr)
        return 1

    failures = []
    for path in records:
        rec = json.loads(path.read_text())
        tree, reported = rec["tree"], rec["error"]

        if tree is None:
            failures.append((path.stem, reported, None, "empty tree"))
            continue

        X, y = load(rec["dataset"])
        preds = [predict_one(tree, x) for x in X]
        if any(p is None for p in preds):
            failures.append((path.stem, reported, None, "reached a leaf with no label"))
            continue

        actual = int((np.array(preds) != y).sum())
        if actual == reported:
            print(f"   {path.stem:42s} error={reported}")
        else:
            failures.append((path.stem, reported, actual, f"tree gives {actual}"))

    for name, reported, _, why in failures:
        print(f"!! {name:42s} reported={reported} but {why}", file=sys.stderr)

    print(
        f"\n{len(records) - len(failures)}/{len(records)} trees reproduce their reported error"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
