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

#[pymodule]
fn pytreesrs(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    odt(py, m)?;
    greed(py, m)?;
    enums(py, m)?;
    Ok(())
}

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

// #[pymodule]
// #[pyo3(name = "rules")]
// fn rules(py: Python<'_>, parent_module:  &Bound<'_, PyModule>) -> PyResult<()> {
//     let module = PyModule::new(py, "rules")?;
//     module.add_class::<ExposedDiscrepancyRule>()?;
//     module.add_class::<ExposedTopKRule>()?;
//     module.add_class::<ExposedPurityRule>()?;
//     module.add_class::<ExposedRestartRule>()?;
//     module.add_class::<ExposedGainRule>()?;
//     parent_module.add_submodule(&module)?;
//     py.import("sys")?
//         .getattr("modules")?
//         .set_item("pytreesrs.enums", module)?;
//     Ok(())
// }

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
