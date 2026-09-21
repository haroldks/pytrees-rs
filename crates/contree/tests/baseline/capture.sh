#!/usr/bin/env bash
# Capture (or re-capture) the behavioural baseline.
#
#   crates/contree/tests/baseline/capture.sh [-s|--smoke] [output-dir]
#
# Runs configurations from matrix.txt through the `baseline` example and writes
# one JSON record per configuration, plus a MANIFEST naming what was attempted.
# Diff the output directory against expected/ with compare.py. The datasets
# are read from $CONTREE_DATASETS, or datasets/ at the repository root.
#
# The full matrix takes ~10 minutes, dominated by a handful of large exhaustive
# searches. --smoke runs the subset tagged `#smoke` in matrix.txt, ~12 seconds,
# which still touches every dataset, every flag combination and every depth.
# Use --smoke while iterating; run the full matrix before committing anything
# that could move a result.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../../.." && pwd)"
bin="$root/target/release/examples/baseline"
datasets="${CONTREE_DATASETS:-$root/datasets}"
timeout_s="${BASELINE_TIMEOUT:-300}"
smoke=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--smoke) smoke=1; shift ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) break ;;
    esac
done
out="${1:-$here/expected}"

if [[ ! -x "$bin" ]]; then
    echo "capture: $bin not built; run: cargo build --release -p contree-rs --example baseline" >&2
    exit 1
fi

mkdir -p "$out"
: > "$out/MANIFEST"
fail=0 ok=0

while IFS= read -r line; do
    # A `#` starts a comment, but a comment may carry tags (`#smoke`) that
    # select which configurations belong to a subset, so read it before
    # stripping it.
    tags=""
    if [[ "$line" == *"#"* ]]; then
        tags="${line#*#}"
        line="${line%%#*}"
    fi
    read -r dataset depth flags <<< "$line"
    [[ -z "${dataset:-}" ]] && continue
    (( smoke )) && [[ "$tags" != *smoke* ]] && continue

    # Flags become part of the filename so the matrix stays self-describing.
    slug="${flags// /}"
    slug="${slug//--/-}"
    name="${dataset}-d${depth}${slug}"
    echo "$name" >> "$out/MANIFEST"

    if timeout "$timeout_s" "$bin" \
        -i "$datasets/${dataset}.txt" -d "$depth" $flags \
        -o "$out/${name}.json" >/dev/null 2>&1
    then
        printf '  ok       %s\n' "$name"
        ok=$((ok + 1))
    else
        rc=$?
        # A timeout here means the matrix is wrong, not that the code is: an
        # entry that cannot finish must not be in the baseline at all.
        [[ $rc -eq 124 ]] && why="TIMEOUT after ${timeout_s}s" || why="exit $rc"
        printf '  FAILED   %s (%s)\n' "$name" "$why" >&2
        rm -f "$out/${name}.json"
        fail=$((fail + 1))
    fi
done < "$here/matrix.txt"

printf '\n%d captured, %d failed -> %s\n' "$ok" "$fail" "$out"
[[ $fail -eq 0 ]]
