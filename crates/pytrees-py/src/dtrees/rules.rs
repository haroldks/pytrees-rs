//! The search rules DL8.5 accepts, read from the plain Python objects in
//! `pytrees.rules`.
//!
//! Each struct pulls its fields off the Python object by attribute name, so
//! the Python side stays an ordinary class that scikit-learn can clone and
//! pickle, and only `fit` turns it into a dtrees rule.

use dtrees_rs::algorithms::optimal::rules::common::TimeLimitRule;
use dtrees_rs::algorithms::optimal::rules::{DiscrepancyRule, GainRule, PurityRule, TopkRule};
use pyo3::prelude::*;

use crate::dtrees::options;

/// `pytrees.rules.DiscrepancyRule`.
#[derive(FromPyObject)]
pub(crate) struct DiscrepancySpec {
    #[pyo3(attribute)]
    initial_value: usize,
    #[pyo3(attribute)]
    limit: Option<usize>,
    #[pyo3(attribute)]
    step_strategy: String,
    #[pyo3(attribute)]
    base: usize,
}

impl DiscrepancySpec {
    pub(crate) fn build(&self) -> PyResult<DiscrepancyRule> {
        let step = options::step_strategy(&self.step_strategy, self.base)?;
        Ok(DiscrepancyRule::new(self.limit.unwrap_or(usize::MAX), step)
            .with_budget(self.initial_value))
    }
}

/// `pytrees.rules.GainRule`.
#[derive(FromPyObject)]
pub(crate) struct GainSpec {
    #[pyo3(attribute)]
    min_gain: f64,
    #[pyo3(attribute)]
    epsilon: f64,
    #[pyo3(attribute)]
    limit: f64,
    #[pyo3(attribute)]
    step_strategy: String,
    #[pyo3(attribute)]
    base: usize,
}

impl GainSpec {
    pub(crate) fn build(&self) -> PyResult<GainRule> {
        let step = options::step_strategy(&self.step_strategy, self.base)?;
        Ok(GainRule::new(self.min_gain, self.epsilon, self.limit, step))
    }
}

/// `pytrees.rules.PurityRule`.
#[derive(FromPyObject)]
pub(crate) struct PuritySpec {
    #[pyo3(attribute)]
    min_purity: f64,
    #[pyo3(attribute)]
    epsilon: f64,
}

impl PuritySpec {
    pub(crate) fn build(&self) -> PurityRule {
        PurityRule::new(self.min_purity, self.epsilon)
    }
}

/// `pytrees.rules.TopKRule`.
#[derive(FromPyObject)]
pub(crate) struct TopKSpec {
    #[pyo3(attribute)]
    initial_value: usize,
    #[pyo3(attribute)]
    limit: Option<usize>,
    #[pyo3(attribute)]
    step_strategy: String,
    #[pyo3(attribute)]
    base: usize,
}

impl TopKSpec {
    pub(crate) fn build(&self) -> PyResult<TopkRule> {
        let step = options::step_strategy(&self.step_strategy, self.base)?;
        Ok(TopkRule::new(self.limit.unwrap_or(usize::MAX), step).with_budget(self.initial_value))
    }
}

/// `pytrees.rules.RestartRule`: restarts the search every `limit` seconds.
#[derive(FromPyObject)]
pub(crate) struct RestartSpec {
    #[pyo3(attribute)]
    limit: f64,
}

impl RestartSpec {
    pub(crate) fn build(&self) -> TimeLimitRule {
        TimeLimitRule::new(self.limit).relaxable()
    }
}
