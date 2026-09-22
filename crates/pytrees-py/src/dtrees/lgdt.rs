//! LGDT, the greedy search over binary features: at each node it keeps the
//! first split of the best depth-2 subtree, then recurses.

use std::time::Instant;

use dtrees_rs::algorithms::common::errors::NativeError;
use dtrees_rs::algorithms::common::types::SearchStatistics;
use dtrees_rs::algorithms::greedy::{LGDTBuilder, LGDT};
use dtrees_rs::algorithms::optimal::depth2::{
    ErrorMinimizer, InfoGainMaximizer, OptimalDepth2Tree,
};
use dtrees_rs::algorithms::TreeSearchAlgorithm;
use dtrees_rs::cover::Cover;
use dtrees_rs::tree::Tree;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::dtrees::data;
use crate::dtrees::options::{self, LgdtCriterion};
use crate::dtrees::output;

/// The LGDT search, as `pytrees.LGDTClassifier` drives it.
#[pyclass(module = "pytrees._native.dtrees")]
pub struct RawLGDT {
    criterion: LgdtCriterion,
    min_sup: usize,
    max_depth: usize,
    fitted: Option<(Tree, SearchStatistics)>,
}

impl RawLGDT {
    fn fitted(&self) -> PyResult<&(Tree, SearchStatistics)> {
        self.fitted
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("this search has not been fitted yet"))
    }
}

/// Fits `learner` and hands back its tree.
fn run<D: OptimalDepth2Tree + Send + ?Sized>(
    py: Python<'_>,
    mut learner: LGDT<D>,
    cover: &mut Cover,
) -> PyResult<Tree> {
    // Release the GIL during the search.
    py.detach(|| learner.fit(cover))
        .map_err(|err| PyRuntimeError::new_err(format!("LGDT failed: {err}")))?;
    Ok(learner.tree().clone())
}

#[pymethods]
impl RawLGDT {
    #[new]
    #[pyo3(signature = (criterion="error", min_sup=1, max_depth=2))]
    fn new(criterion: &str, min_sup: usize, max_depth: usize) -> PyResult<Self> {
        if min_sup == 0 {
            return Err(PyValueError::new_err("min_sup must be greater than 0"));
        }
        if max_depth == 0 {
            return Err(PyValueError::new_err("max_depth must be greater than 0"));
        }
        Ok(Self {
            criterion: options::lgdt_criterion(criterion)?,
            min_sup,
            max_depth,
            fitted: None,
        })
    }

    /// Builds the tree. `y` holds labels encoded as `0..k-1`.
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<()> {
        let mut cover = data::cover(&x, Some(&y))?;
        if cover.count() == 0 {
            return Err(PyValueError::new_err("X has no rows"));
        }
        let invalid = |err: String| PyValueError::new_err(err);
        let start = Instant::now();
        let tree = match self.criterion {
            LgdtCriterion::Error => run(
                py,
                LGDTBuilder::<ErrorMinimizer<NativeError>>::with_default_error_minimizer()
                    .min_support(self.min_sup)
                    .max_depth(self.max_depth)
                    .build()
                    .map_err(invalid)?,
                &mut cover,
            )?,
            LgdtCriterion::InformationGain => run(
                py,
                LGDTBuilder::<InfoGainMaximizer<NativeError>>::with_default_info_gain_maximizer()
                    .min_support(self.min_sup)
                    .max_depth(self.max_depth)
                    .build()
                    .map_err(invalid)?,
                &mut cover,
            )?,
        };
        let statistics = SearchStatistics {
            tree_error: tree.root_error(),
            duration: start.elapsed().as_secs_f64(),
            num_attributes: cover.num_attributes,
            num_samples: cover.num_samples,
            ..Default::default()
        };
        self.fitted = Some((tree, statistics));
        Ok(())
    }

    /// The fitted tree as flat arrays; see `output::tree_arrays`.
    fn tree_arrays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        output::tree_arrays(py, &self.fitted()?.0)
    }

    /// Only the error, duration and sizes: LGDT keeps no search counters.
    #[getter]
    fn statistics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        output::statistics(py, &self.fitted()?.1)
    }

    #[getter]
    fn error(&self) -> PyResult<f64> {
        Ok(self.fitted()?.0.root_error())
    }
}
