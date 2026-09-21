#!/usr/bin/env bash
# One anytime run, recorded as <out-dir>/<label>-<dataset>-<depth>-<method>.json
# with the root incumbent's trajectory: {"error", "status", "trajectory": [[t, e], ...]}.
#
#   run.sh ours     <out-dir> <label> <anytime-binary> <dataset.txt> <depth> <method> [schedule]
#   run.sh upstream <out-dir> <label> <ConTree-binary> <dataset.txt> <depth> <method>
#
# ours:     <anytime-binary> is target/release/examples/anytime (or a copy built
#           at another commit). Methods: contree, contree-gini, lds-first-gini,
#           lds-mid-gini. Schedules: diagonal (default), square.
# upstream: ConTree from https://github.com/ConSol-Lab/contree built with
#           PRINT_INTERMEDIARY_TIME_SOLUTIONS 1 (see README.md). Methods:
#           contree, contree-gini.
#
# The label may not contain '-'. Existing results are kept, so an interrupted
# sweep can be restarted. LIMIT (default 60) sets the time limit in seconds.
set -u
mode=$1 out=$2 label=$3 bin=$4 data=$5 depth=$6 method=$7 schedule=${8:-}
limit=${LIMIT:-60}
dataset=$(basename "$data" .txt)
case $label in *-*) echo "label may not contain '-': $label" >&2; exit 2 ;; esac
mkdir -p "$out"
json=$out/$label-$dataset-$depth-$method.json
[ -f "$json" ] && exit 0

case $mode in
  ours)
    timeout $((limit + 30)) "$bin" "$data" "$depth" "$method" "$limit" $schedule > "$json.tmp" 2>/dev/null
    ;;
  upstream)
    gini=0; [ "$method" = contree-gini ] && gini=1
    timeout $((limit + 60)) "$bin" -file "$data" -max-depth "$depth" -time "$limit" \
        -sort-features-gini-index $gini > "$json.log" 2>&1
    python3 - "$json.log" "$json.tmp" "$limit" <<'PY'
import json, re, sys
log, out, limit = sys.argv[1], sys.argv[2], float(sys.argv[3])
text = open(log).read()
traj = [[float(t), int(round(float(e)))]
        for e, t in re.findall(r"misclassification score ([0-9.]+): ([0-9.]+) seconds", text)]
err = re.search(r"Misclassification score: (\d+)", text)
dur = re.search(r"decision tree: ([0-9.]+) seconds", text)
# Upstream returns before printing when a split reaches error 0 (55c4349 and
# later), so the final tree may be missing from the trajectory.
if err and dur and (not traj or int(err.group(1)) < traj[-1][1]):
    traj.append([max(float(dur.group(1)), traj[-1][0] if traj else 0.0), int(err.group(1))])
if not traj:
    sys.exit(1)
status = "time limit" if dur and float(dur.group(1)) >= limit - 0.1 else "optimal"
json.dump({"error": traj[-1][1], "status": status, "trajectory": traj}, open(out, "w"))
PY
    ;;
  *) echo "mode must be ours or upstream: $mode" >&2; exit 2 ;;
esac

if [ -s "$json.tmp" ]; then mv "$json.tmp" "$json"; else
  rm -f "$json.tmp"; echo "failed: $label $dataset d$depth $method" >&2; exit 1
fi
