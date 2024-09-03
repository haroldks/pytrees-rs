use dtrees_rs::searches::errors::ErrorWrapper;
use dtrees_rs::searches::{Constraints, Statistics};
use dtrees_rs::tree::Tree;
use pyo3::{pyclass, pymethods, PyObject, PyResult, Python};

#[pyclass]
#[derive(Copy, Clone)]
pub(crate) enum ExposedSearchHeuristic {
    InformationGain,
    InformationGainRatio,
    GiniIndex,
    None_,
}

#[pyclass]
#[derive(Copy, Clone)]
pub(crate) enum ExposedDataFormat {
    ClassSupports,
    Tids,
}

#[pyclass]
#[derive(Copy, Clone)]
pub(crate) enum ExposedCacheType {
    Trie,
    Hashmap,
    None_,
}

#[pyclass]
#[derive(Copy, Clone)]
pub(crate) enum ExposedSpecialization {
    Murtree,
    None_,
}

#[pyclass]
#[derive(Copy, Clone)]
pub enum ExposedLowerBoundStrategy {
    Similarity,
    None_,
}

#[pyclass]
#[derive(Copy, Clone)]
pub enum ExposedBranchingStrategy {
    Dynamic,
    None_,
}

#[pyclass]
#[derive(Copy, Clone)]
pub enum ExposedCacheInitStrategy {
    DynamicAllocation,
    UserAllocation,
    None_,
}

#[pyclass]
#[derive(Copy, Clone)]
pub enum ExposedSearchStrategy {
    DiscrepancySearch,
    LessGreedyMurtree,
    LessGreedyInfoGain,
    None_,
}

pub struct PythonError {
    function: PyObject,
}

impl PythonError {
    pub fn new(function: PyObject) -> PythonError {
        PythonError { function }
    }
}

impl ErrorWrapper for PythonError {
    fn compute(&self, data: &[usize]) -> (f64, f64) {
        let mut error = (0., 0.);
        let send_data = data.to_vec();
        Python::with_gil(|py| {
            error = self
                .function
                .call1(py, (send_data,))
                .unwrap()
                .extract(py)
                .unwrap();
        });
        error
    }
}

#[pyclass(name = "Result")]
pub struct LearningResult {
    #[pyo3(get, set)]
    pub(crate) error: f64,
    pub(crate) tree: Tree,
    pub(crate) constraints: Constraints,
    pub(crate) statistics: Statistics,
    pub(crate) duration: f64,
}

#[pymethods]
impl LearningResult {
    // Could be done with paste!

    #[getter]
    pub fn statistics(&self) -> PyResult<String> {
        let json = serde_json::to_string_pretty(&self.statistics).unwrap();
        Ok(json)
    }

    #[getter]
    pub fn constraints(&self) -> PyResult<String> {
        let json = serde_json::to_string_pretty(&self.constraints).unwrap();
        Ok(json)
    }

    #[getter]
    pub fn tree(&self) -> PyResult<String> {
        let json = serde_json::to_string_pretty(&self.tree).unwrap();
        Ok(json)
    }

    #[getter]
    pub fn duration(&self) -> f64 {
        self.duration
    }
}
