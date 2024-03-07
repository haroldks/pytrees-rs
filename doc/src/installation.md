# Installation

Before installing Pytrees-rs, make sure to have [the rust toolchain installed](https://www.rust-lang.org/tools/install).

## Building From Source

```bash
git clone git@github.com:haroldks/pytrees-rs.git
cd pytrees-rs
cargo build --release
```

These commands will create a binary (**dtrees_rs**)in release mode (with compiler optimization). If desired, you can create a symbolic link to this binary in the local binary directory (assuming you are on a Unix-based OS).
```bash
ln -s ./target/release/dtrees_rs $HOME/.local/bin
```


## From crates.io (TBA)
