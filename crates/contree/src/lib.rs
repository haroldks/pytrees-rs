//! Optimal decision trees over continuous features.
//!
//! Two searches are provided:
//!
//! * [`ConTree`](algorithms::ConTree), an exact branch-and-bound search with a
//!   cache of subproblems and a specialised depth-2 solver. Candidate
//!   thresholds lie between consecutive observed values of each feature, and
//!   whole intervals of thresholds are pruned at once (Brită, van der Linden
//!   and Demirović, AAAI 2025).
//! * [`ConTreeLds`](algorithms::ConTreeLds), an anytime version that runs the
//!   same search in passes of growing limited discrepancy budget, so a good
//!   tree is available early (Kiossou, Schaus and Nijssen, *Anytime Optimal
//!   Decision Tree Learning with Continuous Features*, ECML PKDD 2026).
//!
//! Labels must be the integers `0..num_labels`, and a split sends an instance
//! left when `x[feature] <= threshold`, as in scikit-learn.
//!
//! ```
//! use contree::algorithms::ConTree;
//! use contree::common::{PointSelector, SearchConfig};
//! use contree::data::Dataset;
//!
//! // Four instances, one feature, row-major.
//! let values = [0.1, 0.4, 0.6, 0.9];
//! let labels = [0, 0, 1, 1];
//! let dataset = Dataset::from_rows(&values, &labels, 1).unwrap();
//!
//! let config = SearchConfig::new(1, 2, 60.0, 0, usize::MAX, false, true, PointSelector::Mid);
//! let outcome = ConTree::with_config(config).fit(&dataset).unwrap();
//! assert_eq!(outcome.error(), 0);
//! assert_eq!(outcome.tree.predict_one(&[0.7]), Ok(1));
//! ```

// Library code returns text to its caller instead of printing it.
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
    pub fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }
}
