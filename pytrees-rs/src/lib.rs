use crate::greedy::search_lgdt;
use crate::optimal::{optimal_search_dl85, PyGenericDl85};
use crate::utils::{
    ExposedBranchingStrategy, ExposedCacheInitStrategy, ExposedCacheType, ExposedDataFormat,
    ExposedLowerBoundStrategy, ExposedSearchHeuristic, ExposedSearchStrategy,
    ExposedSpecialization,
};
use numpy::pyo3::{pymodule, PyResult, Python};
use pyo3::prelude::PyModule;
use pyo3::wrap_pyfunction;
mod greedy;
mod optimal;
mod utils;

#[pymodule]
fn pytreesrs(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    odt(py, m)?;
    greed(py, m)?;
    enums(py, m)?;
    Ok(())
}

#[pymodule]
#[pyo3(name = "enums")]
fn enums(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let module = PyModule::new(py, "enums")?;
    module.add_class::<ExposedSearchHeuristic>()?;
    module.add_class::<ExposedDataFormat>()?;
    module.add_class::<ExposedCacheType>()?;
    module.add_class::<ExposedSpecialization>()?;
    module.add_class::<ExposedLowerBoundStrategy>()?;
    module.add_class::<ExposedBranchingStrategy>()?;
    module.add_class::<ExposedCacheInitStrategy>()?;
    module.add_class::<ExposedSearchStrategy>()?;

    parent_module.add_submodule(module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("pytreesrs.enums", module)?;
    Ok(())
}

#[pymodule]
#[pyo3(name = "odt")]
fn odt(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let module = PyModule::new(py, "odt")?;
    module.add_function(wrap_pyfunction!(optimal_search_dl85, module)?)?;
    module.add_class::<PyGenericDl85>()?;

    parent_module.add_submodule(module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("pytreesrs.odt", module)?;

    Ok(())
}

#[pymodule]
#[pyo3(name = "greedy")]
fn greed(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let module = PyModule::new(py, "greedy")?;
    module.add_function(wrap_pyfunction!(search_lgdt, module)?)?;

    parent_module.add_submodule(module)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("pytreesrs.greedy", module)?;

    Ok(())
}
