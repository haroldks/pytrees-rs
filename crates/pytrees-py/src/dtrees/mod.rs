//! Bindings for the dtrees searches over binary features: DL8.5 and LGDT.

mod data;
mod dl85;
mod errors;
mod lgdt;
mod options;
mod output;
mod rules;

use pyo3::prelude::*;

/// Fills `pytrees._native.dtrees`.
pub fn add_classes(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<dl85::RawDL85>()?;
    module.add_class::<lgdt::RawLGDT>()?;
    Ok(())
}
