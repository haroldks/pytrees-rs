# Installation

## From PyPI

```bash
pip install pytrees-rs
```

The package needs Python 3.10 or later, and installs NumPy, SciPy and
scikit-learn. Prebuilt wheels are published for Linux (x86_64 and aarch64),
macOS (Intel and Apple silicon) and Windows (x64). The package is imported as
`pytrees`:

```python
import pytrees
from pytrees import ConTreeClassifier, DL85Classifier, LGDTClassifier, DL85Cluster
```

## From source

Building from source needs a Rust toolchain, version 1.77 or later. Install it
with [rustup](https://rustup.rs) if you do not have one. Then:

```bash
git clone https://github.com/haroldks/pytrees-rs.git
cd pytrees-rs
pip install .
```

`pip` builds the Rust extension with [maturin](https://www.maturin.rs), which
it installs on its own.

For development, build the package in place instead, so that Python changes
are picked up without reinstalling:

```bash
pip install maturin
maturin develop --release
pytest python/tests
```

## Command line tools

The command line tools are plain Rust binaries:

```bash
cargo build --release -p dtrees-cli -p contree-cli
./target/release/dtrees --help
./target/release/con-tree --help
```

See [Command line tools](cli.md) for their options.
