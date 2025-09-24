pub mod builder;

use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use crate::common::create_cover_from_numpy;
use crate::common::enums::ExposedSearchStrategy;
use crate::common::types::SearchOutput;

/// Less Greedy Decision Tree (LGDT) construction function.
///
/// This function implements greedy decision tree construction algorithms that provide
/// fast approximate solutions for decision tree learning. Unlike optimal algorithms
/// like DL8.5, greedy approaches make locally optimal choices at each step, resulting
/// in faster execution times at the cost of global optimality.
///
//
///
/// # Parameters
///
/// - `input`: Feature matrix as NumPy array of shape (n_samples, n_features)
/// - `target`: Target vector as NumPy array of shape (n_samples,)
/// - `search_strategy`: The search strategy to use for tree construction
/// - `min_sup`: Minimum support (number of samples) required for a split
/// - `max_depth`: Maximum depth allowed for the constructed tree
///
/// # Returns
///
/// A `SearchOutput` object containing:
/// - `error`: Classification error of the constructed tree
/// - `tree`: JSON representation of the decision tree structure
/// - `statistics`: Search statistics (nodes created, splits evaluated, etc.)
/// - `duration`: Total construction time in seconds
///
/// # Errors
///
/// Returns `PyValueError` if:
/// - `min_sup` is 0 (must be at least 1)
/// - `max_depth` is 0 (must be at least 1)
/// - Input data contains no samples
/// - Input arrays have incompatible shapes
/// - Memory allocation fails during tree construction
///
/// # Example
///
/// ```python
/// import numpy as np
/// from pytreesrs.greedy import lgdt
/// from pytreesrs.enums import ExposedSearchStrategy
///
/// # Generate sample data
/// X = np.random.rand(1000, 10)
/// y = np.random.randint(0, 3, 1000)
///
/// # Build greedy decision tree
/// result = lgdt(
///     input=X,
///     target=y,
///     search_strategy=ExposedSearchStrategy.LGDTErrorMinimizer,
///     min_sup=10,
///     max_depth=5
/// )
///
/// print(f"Tree error: {result.error}")
/// print(f"Construction time: {result.duration}s")
/// print(f"Tree structure: {result.tree}")
/// ```
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

    // Validate data
    if cover.count() == 0 {
        return Err(PyValueError::new_err("Input data contains no samples"));
    }

    // Build and execute the LGDT algorithm
    let mut builder = search_strategy.to_lgdt_builder(min_sup, max_depth)?;

    builder.fit_and_get_result(&mut cover)
}
