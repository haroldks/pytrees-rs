mod core;
mod manager;

// Re-export public API
pub use core::Reason;
pub use core::{Rule, RuleContext, RuleResult, RuleState};
pub use manager::RuleManager;

// Common rules that can be used with any tree learning algorithm
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
