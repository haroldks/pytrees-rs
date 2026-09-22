//! The string options the Python estimators take, parsed into dtrees types.
//!
//! Options are plain strings so that scikit-learn can clone, pickle and
//! compare them. Each parser lists the accepted values when it rejects one.

use dtrees_rs::algorithms::common::heuristics::{
    GiniIndex, Heuristic, InformationGain, NoHeuristic, WeightedEntropy,
};
use dtrees_rs::algorithms::common::types::NodeDataType;
use dtrees_rs::algorithms::optimal::rules::{Exponential, Luby, Monotonic, StepStrategy};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

fn unknown(option: &str, value: &str, accepted: &str) -> pyo3::PyErr {
    PyValueError::new_err(format!(
        "unknown {option} {value:?}; expected one of {accepted}"
    ))
}

/// `heuristic`: the order in which DL8.5 tries the features.
pub(crate) fn heuristic(value: &str) -> PyResult<Box<dyn Heuristic>> {
    Ok(match value {
        "none" => Box::new(NoHeuristic),
        "gini" => Box::new(GiniIndex),
        "information_gain" => Box::<InformationGain>::default(),
        "weighted_entropy" => Box::new(WeightedEntropy),
        _ => {
            return Err(unknown(
                "heuristic",
                value,
                "none, gini, information_gain, weighted_entropy",
            ))
        }
    })
}

/// `error_function_input`: what a custom error function receives at each
/// node, the class counts or the indices of the rows in it.
pub(crate) fn error_function_input(value: &str) -> PyResult<NodeDataType> {
    match value {
        "class_counts" => Ok(NodeDataType::ClassesSupport),
        "indices" => Ok(NodeDataType::Tids),
        _ => Err(unknown(
            "error_function_input",
            value,
            "class_counts, indices",
        )),
    }
}

/// `step_strategy`: how a rule's budget grows between restarts.
pub(crate) fn step_strategy(value: &str, base: usize) -> PyResult<Box<dyn StepStrategy>> {
    Ok(match value {
        "monotonic" => Box::new(Monotonic::new(base)),
        "exponential" => Box::new(Exponential::new(base)),
        "luby" => Box::new(Luby::new(base)),
        _ => {
            return Err(unknown(
                "step_strategy",
                value,
                "monotonic, exponential, luby",
            ))
        }
    })
}

/// `criterion` of LGDT: what its depth-2 lookahead optimises.
#[derive(Clone, Copy)]
pub(crate) enum LgdtCriterion {
    Error,
    InformationGain,
}

pub(crate) fn lgdt_criterion(value: &str) -> PyResult<LgdtCriterion> {
    match value {
        "error" => Ok(LgdtCriterion::Error),
        "information_gain" => Ok(LgdtCriterion::InformationGain),
        _ => Err(unknown("criterion", value, "error, information_gain")),
    }
}
