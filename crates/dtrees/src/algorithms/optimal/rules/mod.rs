//! Rules that decide, at each node, whether the DL8.5 search goes on.
//!
//! The search consults two sets of rules. *Node rules* ([`common`]) encode the
//! problem: maximum depth, minimum support, pure nodes and bounds. *Search
//! rules* restrict the search to make it anytime: they cut branches such as
//! the ones beyond a discrepancy budget ([`DiscrepancyRule`], LDS-DL8.5) or
//! beyond the `k` best-ranked features ([`TopkRule`], Top-k-DL8.5).
//!
//! When a relaxable rule has cut part of the search, the search is restarted
//! with every rule relaxed (its budget widened), until no rule cuts anything
//! and the tree is optimal. This is the CA-DL8.5 framework of Kiossou and
//! Schaus, *A Generic Complete Anytime Beam Search for Optimal Decision Trees*
//! (IDA 2026).

mod core;
mod manager;

pub use core::Reason;
pub use core::{Rule, RuleContext, RuleResult, RuleState};
pub use manager::RuleManager;

pub mod common;
mod discrepancy;
mod gain;
mod helpers;
mod purity;
mod topk;

pub use discrepancy::DiscrepancyRule;
pub use gain::GainRule;
pub use helpers::*;
pub use purity::PurityRule;
pub use topk::{DecreasingTopkRule, TopkRule};
