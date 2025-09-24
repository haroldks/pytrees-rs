//! # pytrees-rs: Python Bindings for Decision Tree Algorithms
//!
//! This crate provides Python bindings for the `dtrees-rs` library, exposing
//! decision tree algorithms through a PyO3-based interface. The library focuses on optimal
//! decision tree construction using advanced search strategies and rule-based optimization.
//!
//! ## Module Structure
//!
//! The Python wrapper is organized into several key modules:
//!
//! - **`odt`**: Optimal Decision Trees module containing the DL8.5 algorithm implementation
//! - **`greedy`**: Greedy algorithms module with LGDT (Local Greedy Decision Tree) variants
//! - **`enums`**: Configuration enumerations for algorithms, heuristics, and policies
//!
//! ## Key Features
//!
//! - **DL8.5 Algorithm**: State-of-the-art optimal decision tree construction
//! - **Rule-based Optimization**: Support for discrepancy, gain, purity, and top-k rules
//! - **Multiple Heuristics**: Information gain, Gini index, weighted entropy
//! - **Flexible Configuration**: Extensive parameterization for different use cases
//! - **Modern Cover API**: Efficient data handling with zero-copy optimizations
//!
//! ## Example Usage
//!
//! ```python
//! import pytreesrs
//! from pytreesrs.enums import ExposedHeuristic
//! from pytreesrs.odt.rules import ExposedGainRule
//!
//! # Create DL8.5 classifier with information gain heuristic
//! classifier = pytreesrs.odt.PyDL85(
//!     max_depth=3,
//!     min_sup=5,
//!     heuristic=ExposedHeuristic.InformationGain,
//!     gain=ExposedGainRule(min_gain=0.01)
//! )
//!
//! # Fit the model
//! classifier.fit(X_train, y_train)
//!
//! # Get results
//! stats = classifier.stats
//! print(f"Training error: {stats.error}")
//! print(f"Tree structure: {stats.tree}")
//! ```

use crate::greedy::search_lgdt;
use numpy::pyo3::{pymodule, PyResult, Python};
use pyo3::prelude::{PyModule, PyModuleMethods, Bound, PyAnyMethods};
use pyo3::wrap_pyfunction;
use crate::common::enums::{ExposedBranchingPolicy, ExposedCacheInitStrategy, ExposedCacheType, ExposedDepth2Policy, ExposedHeuristic, ExposedLowerBoundPolicy, ExposedNodeDataType, ExposedSearchStrategy, ExposedStepStrategy};
use crate::common::types::{ExposedDiscrepancyRule, ExposedGainRule, ExposedPurityRule, ExposedRestartRule, ExposedTopKRule};
use crate::optimal::PyDL85;

mod greedy;
mod optimal;
mod common;

/// PyO3 module entry point for pytreesrs.
///
/// This function initializes all submodules and makes them available to Python.
/// The module structure follows a hierarchical organization:
///
/// - `pytreesrs.odt`: Optimal decision tree algorithms
/// - `pytreesrs.greedy`: Greedy decision tree algorithms
/// - `pytreesrs.enums`: Configuration enumerations
#[pymodule]
fn pytreesrs(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    odt(py, m)?;
    greed(py, m)?;
    enums(py, m)?;
    Ok(())
}

/// Enumerations submodule containing configuration types.
///
/// This module exposes various enumeration types used to configure
/// algorithm behavior, heuristics, and optimization policies.
///
/// Available enums:
/// - `ExposedHeuristic`: Heuristic functions (NoHeuristic, GiniIndex, InformationGain, etc.)
/// - `ExposedNodeDataType`: Node data representation strategies
/// - `ExposedCacheType`: Caching mechanisms for search optimization
/// - `ExposedDepth2Policy`: Depth-2 specialization policies
/// - `ExposedLowerBoundPolicy`: Lower bound computation strategies
/// - `ExposedBranchingPolicy`: Branching strategies for tree construction
/// - `ExposedCacheInitStrategy`: Cache initialization approaches
/// - `ExposedSearchStrategy`: Search strategies for greedy algorithms
/// - `ExposedStepStrategy`: Step strategies for rule-based optimization
#[pymodule]
#[pyo3(name = "enums")]
fn enums(py: Python<'_>, parent_module:  &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new(py, "enums")?;
    module.add_class::<ExposedHeuristic>()?;
    module.add_class::<ExposedNodeDataType>()?;
    module.add_class::<ExposedCacheType>()?;
    module.add_class::<ExposedDepth2Policy>()?;
    module.add_class::<ExposedLowerBoundPolicy>()?;
    module.add_class::<ExposedBranchingPolicy>()?;
    module.add_class::<ExposedCacheInitStrategy>()?;
    module.add_class::<ExposedSearchStrategy>()?;
    module.add_class::<ExposedStepStrategy>()?;

    parent_module.add_submodule(&module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("pytreesrs.enums", module)?;
    Ok(())
}

/// Optimal Decision Trees (ODT) submodule.
///
/// This module provides access to optimal decision tree algorithms,
/// primarily the DL8.5 algorithm with comprehensive rule support.
///
/// Main components:
/// - `PyDL85`: The main optimal decision tree classifier
/// - `rules`: Submodule containing rule-based optimization classes
///
/// The rules submodule includes:
/// - `ExposedDiscrepancyRule`: Limited discrepancy search rules
/// - `ExposedGainRule`: Information gain-based stopping rules
/// - `ExposedPurityRule`: Node purity-based rules
/// - `ExposedTopKRule`: Top-k search limitation rules
/// - `ExposedRestartRule`: Time-based restart rules
#[pymodule]
#[pyo3(name = "odt")]
fn odt(py: Python<'_>, parent_module:  &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new(py, "odt")?;
    module.add_class::<PyDL85>()?;


    let rules_module = PyModule::new(py, "rules")?;
    rules_module.add_class::<ExposedDiscrepancyRule>()?;
    rules_module.add_class::<ExposedTopKRule>()?;
    rules_module.add_class::<ExposedPurityRule>()?;
    rules_module.add_class::<ExposedRestartRule>()?;
    rules_module.add_class::<ExposedGainRule>()?;

    module.add_submodule(&rules_module)?;
    parent_module.add_submodule(&module)?;



    py.import("sys")?
        .getattr("modules")?
        .set_item("pytreesrs.odt", module)?;

    py.import("sys")?
        .getattr("modules")?
        .set_item("pytreesrs.odt.rules", rules_module)?;

    Ok(())
}

/// Greedy algorithms submodule.
///
/// This module provides access to greedy decision tree construction
/// algorithms, specifically variants of the LGDT algorithm.
///
/// Available functions:
/// - `lgdt`: Less Greedy Decision Tree construction with configurable search strategies
///
/// The greedy algorithms offer faster tree construction compared to optimal
/// methods, making them suitable for large datasets or when approximate
/// solutions are acceptable.
#[pymodule]
#[pyo3(name = "greedy")]
fn greed(py: Python<'_>, parent_module:  &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new(py, "greedy")?;
    module.add_function(wrap_pyfunction!(search_lgdt, &module)?)?;

    parent_module.add_submodule(&module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("pytreesrs.greedy", module)?;

    Ok(())
}
