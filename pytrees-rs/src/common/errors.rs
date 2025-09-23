use pyo3::{Py, PyAny, Python};
use dtrees_rs::algorithms::common::errors::ErrorWrapper;

pub struct PythonError {
    function: Py<PyAny>,
}

impl PythonError {
    pub fn new(function: Py<PyAny>) -> PythonError {
        PythonError { function }
    }
}

impl ErrorWrapper for PythonError {
    fn compute(&self, data: &[usize]) -> (f64, f64) {
        let mut error = (0., 0.);
        let send_data = data.to_vec();
        Python::attach(|py| {
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
