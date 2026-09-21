//! Optimal decision trees over continuous features.
//!
//! The search is a DL8.5-style branch and bound with a bitset-keyed cache,
//! specialized at depth two, over candidate split points drawn from the
//! observed values of each feature.
//!
//! Two conventions run through the whole crate and are easy to get wrong:
//!
//! * **The label lives in column 0** of an input file, and labels must be the
//!   dense integers `0..num_labels`.
//! * **A split routes left when `x[feature] <= threshold`**, right otherwise,
//!   as in scikit-learn.

// A library hands text back to its caller rather than printing it; the
// binaries and examples print. Tests may print.
#![cfg_attr(not(test), warn(clippy::print_stdout, clippy::print_stderr))]

pub mod algorithms;
mod bitsets;
mod caching;
pub mod common;
pub mod data;
pub mod reader;
pub mod tree;

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    /// Absolute path to a file in `crates/contree/tests/fixtures/`.
    ///
    /// Resolved from `CARGO_MANIFEST_DIR` rather than the working directory, so
    /// the tests keep working wherever cargo is invoked from.
    pub fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }
}
