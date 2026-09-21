//! `pytrees._native`: the compiled half of the `pytrees` package.
//!
//! The submodules are thin: they turn numpy arrays into each library's data
//! type, run the search, and hand the result back. Everything users see --
//! parameters, validation, scikit-learn behaviour -- lives in the Python
//! package around this module.
//!
//! - `pytrees._native.odt`: DL8.5, the optimal search on binary features
//! - `pytrees._native.greedy`: LGDT, the greedy search on binary features
//! - `pytrees._native.contree`: ConTree, the optimal search on continuous features

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::dtrees::dl85::PyDL85;
use crate::dtrees::lgdt::search_lgdt;

mod contree;
mod dtrees;

#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let odt = PyModule::new(py, "odt")?;
    odt.add_class::<PyDL85>()?;
    add_submodule(py, m, &odt)?;

    let greedy = PyModule::new(py, "greedy")?;
    greedy.add_function(wrap_pyfunction!(search_lgdt, &greedy)?)?;
    add_submodule(py, m, &greedy)?;

    contree::register(py, m)?;
    Ok(())
}

/// Adds `module` under `parent` and registers it in `sys.modules`, which is
/// what makes `from pytrees._native.<name> import ...` work.
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
