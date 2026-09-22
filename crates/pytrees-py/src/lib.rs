//! `pytrees._native`: the compiled half of the `pytrees` package.
//!
//! Each submodule turns numpy arrays into a library's data type, runs the
//! search, and returns the result. Parameters, validation and the
//! scikit-learn interface live in the Python package around this module.
//!
//! - `pytrees._native.dtrees`: `RawDL85` and `RawLGDT`, over binary features
//! - `pytrees._native.contree`: `RawConTree`, over continuous features
//! - `pytrees._native.tree`: `apply`, which every estimator's `Tree` uses

// Library code returns text to its caller instead of printing it.
#![cfg_attr(not(test), warn(clippy::print_stdout, clippy::print_stderr))]

use pyo3::prelude::*;

mod contree;
mod dtrees;
mod tree;

#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new(py, "dtrees")?;
    dtrees::add_classes(&module)?;
    add_submodule(py, m, &module)?;

    let module = PyModule::new(py, "contree")?;
    contree::add_classes(&module)?;
    add_submodule(py, m, &module)?;

    let module = PyModule::new(py, "tree")?;
    module.add_function(wrap_pyfunction!(tree::apply, &module)?)?;
    add_submodule(py, m, &module)?;
    Ok(())
}

/// Adds `module` under `parent` and registers it in `sys.modules`. That is
/// what makes `from pytrees._native.<name> import ...` work, and what lets
/// pickle find a class by the module name its `#[pyclass]` declares.
fn add_submodule(
    py: Python<'_>,
    parent: &Bound<'_, PyModule>,
    module: &Bound<'_, PyModule>,
) -> PyResult<()> {
    parent.add_submodule(module)?;
    let name = format!("pytrees._native.{}", module.name()?);
    py.import("sys")?
        .getattr("modules")?
        .set_item(name, module)?;
    Ok(())
}
