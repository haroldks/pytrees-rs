#!/usr/bin/env python3
"""Diff a freshly captured baseline against the frozen expected one.

    crates/contree/tests/baseline/compare.py <new-dir> [expected-dir]

Reports three things separately, because they mean different things:

  * changed error   - the search found a different-quality tree. During this
                      refactor some of these are expected (the A1/A3 fixes
                      change pruning), so each one must be accounted for.
  * changed tree    - same error, different tree. Usually a tie broken
                      differently; benign, but worth seeing.
  * changed counters- same result, different amount of work. Pure-performance
                      changes land here and nowhere else.

A capture written by capture.sh carries a MANIFEST of the configurations it
attempted, so comparing a `--smoke` subset reports only what that subset
covered; the rest is counted as skipped, not as missing.
"""

import json
import pathlib
import sys


def load(directory: pathlib.Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))}


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2

    new_dir = pathlib.Path(argv[1])
    new = load(new_dir)
    old = load(
        pathlib.Path(argv[2])
        if len(argv) > 2
        else pathlib.Path(__file__).resolve().parent / "expected"
    )

    # Restrict to what this capture actually set out to produce, so that a
    # subset run does not accuse the untouched configurations of vanishing.
    manifest = new_dir / "MANIFEST"
    if manifest.is_file():
        attempted = {n for n in manifest.read_text().split() if n}
        skipped = sorted(set(old) - attempted)
        old = {k: v for k, v in old.items() if k in attempted}
    else:
        skipped = []

    missing = sorted(set(old) - set(new))
    added = sorted(set(new) - set(old))
    errors, trees, counters = [], [], []

    for name in sorted(set(old) & set(new)):
        a, b = old[name], new[name]
        if a["error"] != b["error"]:
            errors.append((name, a["error"], b["error"]))
        elif a["tree"] != b["tree"]:
            trees.append(name)
        elif a["counters"] != b["counters"]:
            counters.append((name, a["counters"], b["counters"]))

    for name in missing:
        print(f"!! MISSING   {name} (was captured before, is not now)")
    for name in added:
        print(f"   added     {name}")
    for name, before, after in errors:
        direction = "better" if after < before else "WORSE"
        print(f"!! ERROR     {name}: {before} -> {after}  ({direction})")
    for name in trees:
        print(f"   tree      {name}: same error, different tree")
    for name, before, after in counters:
        deltas = ", ".join(
            f"{k}: {before[k]}->{after[k]}" for k in before if before[k] != after[k]
        )
        print(f"   counters  {name}: {deltas}")

    unchanged = len(set(old) & set(new)) - len(errors) - len(trees) - len(counters)
    tail = f", {len(skipped)} not in this subset" if skipped else ""
    print(
        f"\n{unchanged} identical, {len(counters)} counters-only, "
        f"{len(trees)} tree-only, {len(errors)} error changes, "
        f"{len(missing)} missing, {len(added)} added{tail}"
    )
    return 1 if (errors or missing) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
