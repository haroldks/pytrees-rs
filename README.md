# CADL8.5: Complete Anytime DL8.5

CADL8.5 is a **complete anytime framework** for decision tree learning that extends **DL8.5** with **rule-based restart strategies** inspired by **Complete Anytime Beam Search (CABS)**.
It quickly finds high-quality solutions and improves them over time, while still guaranteeing **optimal solutions**.
Compared to DL8.5 and Blossom, CADL8.5 often solves **more instances to optimality** and performs at least as well as greedy baselines in early stages.

---

## Installation and Execution

This project is implemented in **Rust**, and `cargo` is required to build and run it.
Clone and build:

```bash
git clone <repository-url>
cd CADL8.5
cargo build --release
```

All algorithms are provided as examples in the examples/ folder. You can run them as follows:

```bash
cargo run --release --example $ALGO -- \
    --input $DATASET \
    --depth $DEPTH \
    --support $SUPPORT \
    --timeout $TIMEOUT \
    --heuristic information-gain \
    --result $OUTPUT_DIR \
    --fast-d2 enabled
```

Where:

$ALGO is the example file name (e.g., discrepancy, topk, gain, ...)

$DATASET is the dataset path (.txt)

$DEPTH is the maximum decision tree depth

$SUPPORT is the minimum support threshold (integer)

$TIMEOUT is the time limit in seconds

$OUTPUT_DIR is the folder where results are stored

**Example** :

```bash
cargo run --release --example discrepancy -- \
    --input test_data/anneal.txt \
    --depth 6 \
    --support 1 \
    --timeout 300 \
    --result .
    --fast-d2 enabled

```
