# Installation

This guide will help you install pytrees-rs on your system. We provide multiple installation methods to suit different needs and environments.

## Prerequisites

Before installing pytrees-rs, ensure you have the following:

### Python Requirements
- **Python 3.8 or higher** (Python 3.9+ recommended)
- **pip** package manager
- **NumPy** (automatically installed with pytrees-rs)

### For Source Installation
- **Rust toolchain** (1.70.0 or higher)
- **Git** (for cloning the repository)

### Installing Rust

If you need to install Rust, use the official installer:

```bash
# Install Rust using rustup (recommended)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Restart your shell or run:
source ~/.cargo/env

# Verify installation
rustc --version
cargo --version
```

**Alternative methods:**
- **macOS**: `brew install rust`
- **Ubuntu/Debian**: `sudo apt install rustc cargo`
- **Windows**: Download from [rustup.rs](https://rustup.rs/)

## Installation Methods

### Method 1: PyPI Installation

```bash
pip install pytrees-rs
```

### Method 2: Building from Source (Current Method)

This is currently the primary installation method:

```bash
# Clone the repository
git clone https://github.com/haroldks/pytrees-rs.git
cd pytrees-rs

# Build the Rust binary
cargo build --release

# Install Python package
cd pytrees-rs  # Navigate to Python package directory
pip install .
```

### Method 3: Binary Installation Only

If you only need the command-line interface:

```bash
# Clone and build
git clone https://github.com/haroldks/pytrees-rs.git
cd pytrees-rs
cargo build --release

# Create symbolic link (Unix-based systems)
ln -s $(pwd)/target/release/dtrees_rs $HOME/.local/bin/dtrees_rs

# Or copy to system path
sudo cp target/release/dtrees_rs /usr/local/bin/
```

## Basic Usage

After installation, verify that PyTrees-RS is working:

### Python Library Test

```python
# Test basic import
import pytrees
from pytrees import DL85Classifier, LGDTClassifier

# Quick functionality test
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=5, random_state=42)

clf = DL85Classifier(max_depth=2, min_sup=5)
clf.fit(X, y)
print(f"Accuracy: {clf.score(X, y):.3f}")
```

### Command Line Test

```bash
# Test binary installation
dtrees_rs --help
```

## Next Steps

Once installed successfully:

1. **[Quick Start Guide](./quickstart.md)**: Build your first optimal tree
2. **[Python Library](./python/README.md)**: Explore the API
3. **[Examples](./examples/classification.md)**: See real applications


