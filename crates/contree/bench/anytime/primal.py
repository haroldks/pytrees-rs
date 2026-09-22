"""Average primal gap of anytime runs recorded by run.sh.

Definitions:

    gap(e)      = 0 if e == best, else |e - best| / max(e, best)
    gap(t)      = 1 until the first solution, then the gap of the incumbent
    P_ratio(T)  = (integral of gap(t) over [0, T]) / T

`best` is the lowest error any run reached on that (dataset, depth), over
every run given and, with --reference, every result in that CSV (columns
name, depth, error).

    python3 primal.py <run-dir>... [--reference all.csv] [--times 5,15,30,60] [--detail]

Rows are (label, method); a row's `n` is how many datasets it has a run for,
so compare rows with equal `n`.
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def gap(e, best):
    return 0.0 if e == best else abs(e - best) / max(e, best)


def p_ratio(trajectory, best, T):
    t_prev, g_prev, area = 0.0, 1.0, 0.0
    for t, e in trajectory:
        if t > T:
            break
        area += g_prev * (t - t_prev)
        t_prev, g_prev = t, gap(e, best)
    return (area + g_prev * (T - t_prev)) / T


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--reference")
    ap.add_argument("--times", default="5,15,30,60")
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()
    times = [float(t) for t in args.times.split(",")]

    runs = {}  # (label, dataset, depth, method) -> trajectory
    for d in args.dirs:
        for f in glob.glob(os.path.join(d, "*.json")):
            label, dataset, depth, method = os.path.basename(f)[:-5].split("-", 3)
            runs[(label, dataset, int(depth), method)] = [
                tuple(x) for x in json.load(open(f))["trajectory"]
            ]

    best = defaultdict(lambda: float("inf"))
    for (_, dataset, depth, _), traj in runs.items():
        for _, e in traj:
            best[(dataset, depth)] = min(best[(dataset, depth)], e)
    if args.reference:
        for row in csv.DictReader(open(args.reference)):
            key = (row["name"], int(row["depth"]))
            if key in best and row["error"]:
                best[key] = min(best[key], int(float(row["error"])))

    table = defaultdict(
        list
    )  # (depth, label, method) -> [[P_ratio at each T] per dataset]
    for (label, dataset, depth, method), traj in sorted(runs.items()):
        table[(depth, label, method)].append(
            [p_ratio(traj, best[(dataset, depth)], T) for T in times]
        )

    for depth in sorted({k[0] for k in table}):
        print(f"\n=== depth {depth}: average primal gap (%), lower is better")
        print(
            f"{'label':<14}{'method':<16}{'n':>4}"
            + "".join(f"{f'{T:g}s':>8}" for T in times)
        )
        for (k, label, method), rows in sorted(table.items()):
            if k == depth:
                means = [
                    100 * sum(r[i] for r in rows) / len(rows) for i in range(len(times))
                ]
                print(
                    f"{label:<14}{method:<16}{len(rows):>4}"
                    + "".join(f"{m:8.1f}" for m in means)
                )

    if args.detail:
        T = times[-1]
        print(f"\n=== per dataset: P_ratio at {T:g}s (%) and final error")
        for dataset, depth in sorted(best):
            print(f"\n{dataset} d{depth}  best={best[(dataset, depth)]}")
            for (label, d, k, method), traj in sorted(runs.items()):
                if (d, k) == (dataset, depth):
                    final = traj[-1][1] if traj else "-"
                    print(
                        f"  {label:<14}{method:<16}{100 * p_ratio(traj, best[(d, k)], T):6.1f}%  error={final}"
                    )


if __name__ == "__main__":
    main()
