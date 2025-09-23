pub mod builder;

use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use crate::common::create_cover_from_numpy;
use crate::common::enums::ExposedSearchStrategy;
use crate::common::types::SearchOutput;

#[pyfunction]
#[pyo3(name = "lgdt")]
pub(crate) fn search_lgdt(
    input: PyReadonlyArrayDyn<f64>,
    target: Option<PyReadonlyArrayDyn<f64>>,
    search_strategy: ExposedSearchStrategy,
    min_sup: usize,
    max_depth: usize,
) -> PyResult<SearchOutput> {
    if min_sup == 0 {
        return Err(PyValueError::new_err("min_sup must be greater than 0"));
    }
    if max_depth == 0 {
        return Err(PyValueError::new_err("max_depth must be greater than 0"));
    }

    let mut cover = create_cover_from_numpy(input, target.as_ref())?;

    if cover.count() == 0 {
        return Err(PyValueError::new_err("Input data contains no samples"));
    }

    let mut builder = search_strategy.to_lgdt_builder(min_sup, max_depth)?;

    builder.fit_and_get_result(&mut cover)
}

