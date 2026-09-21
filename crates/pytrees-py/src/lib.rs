//! `pytrees._native`: the compiled half of the `pytrees` package.
//!
//! One submodule per library, both thin: they turn numpy arrays into the
//! library's data type, run the search, and hand the result back. Everything
//! users see -- parameters, validation, scikit-learn behaviour -- lives in
//! the Python package around this module.
//!
//! - `pytrees._native.dtrees`: `RawDL85` and `RawLGDT`, over binary features
//! - `pytrees._native.contree`: `RawConTree`, over continuous features
//! - `pytrees._native.tree`: `apply`, which every estimator's `Tree` uses

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
