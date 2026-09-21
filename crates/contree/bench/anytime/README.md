# Anytime benchmarks

How the anytime numbers in the commit messages were produced: runs with a
60 s limit, scored by the average primal gap of the notebook in the
contree-rs repository's `experiments/`. Commands run from the repository root.

- `run.sh` — one run, recorded as `<label>-<dataset>-<depth>-<method>.json`
  (the root incumbent's `(seconds, error)` trajectory).
- `primal.py` — average primal gap per `(label, method)` and depth over a set
  of run directories.

## Solvers

**This crate.**

```sh
cargo build --release -p contree-rs --example anytime
```

To compare against another commit, build it in a worktree and give its binary
its own label:

```sh
git worktree add ../contree-at-X <commit>
(cd ../contree-at-X && cargo build --release -p contree-rs --example anytime)
cp ../contree-at-X/target/release/examples/anytime bench-bin/anytime-X
```

**Upstream ConTree**, printing at every root improvement. The paper's baseline
logs printed once per completed root feature instead, which overstates
ConTree's gap. The runs here used 55c4349 (the paper's version) and 61ebd49
(with the upper-bound fixes).

```sh
git clone https://github.com/ConSol-Lab/contree && cd contree && git checkout 61ebd49
sed -i 's/#define PRINT_INTERMEDIARY_TIME_SOLUTIONS 0/#define PRINT_INTERMEDIARY_TIME_SOLUTIONS 1/' \
    code/Utilities/include/configuration.h
cmake -S code -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

## A sweep

Datasets used so far:

- depth 4: `bank raisin wilt page segment rice`
- depth 5: `avila bank bean bidding eeg fault htru magic occupancy page raisin rice room segment skin wilt`

The depth-5 files came from upstream's `test_data/`. The runs are
independent, so run them in parallel, but on fewer workers than cores, since
they are timed:

```sh
out=target/bench-anytime
{
  for d in bank raisin wilt page segment rice; do
    for m in lds-first-gini lds-mid-gini; do
      echo ours $out head target/release/examples/anytime datasets/$d.txt 4 $m diagonal
    done
    echo upstream $out up61 ../contree/build/ConTree datasets/$d.txt 4 contree-gini
  done
} | xargs -P 4 -L 1 crates/contree/bench/anytime/run.sh

python3 crates/contree/bench/anytime/primal.py $out --reference <contree-rs>/experiments/results_27122025/all.csv
```

`primal.py --detail` adds the per-dataset breakdown. The noise between two
sweeps of the same binary is about 0.5 points of average gap.

## Pass-by-pass diagnostics

`cargo run --release -p contree-rs --example lds_passes -- <dataset.txt> <depth> <first|mid> <limit> [schedule] [max-passes]`
prints each pass's error, budget, cache size and solver calls. With a fixed
`max-passes`, the counters give a comparison of work done that doesn't depend
on timing.
