use std::sync::{Arc, Mutex};

use dtrees_rs::algorithms::common::errors::ErrorWrapper;
use pyo3::{Py, PyAny, PyErr, Python};

/// Where a `PythonError` leaves the first exception the caller's function
/// raised. The search cannot stop mid-way, so the owner checks this once it
/// returns and re-raises what it finds.
pub type ErrorSlot = Arc<Mutex<Option<PyErr>>>;

/// A caller's Python error function, seen by the search as an `ErrorWrapper`.
pub struct PythonError {
    function: Py<PyAny>,
    failure: ErrorSlot,
}

impl PythonError {
    pub fn new(function: Py<PyAny>, failure: ErrorSlot) -> PythonError {
        PythonError { function, failure }
    }
}

/// Returned in place of a real error once the function has failed: it makes
/// every node look useless, so the search winds down quickly.
const FAILED: (f64, f64) = (f64::INFINITY, 0.0);

impl ErrorWrapper for PythonError {
    fn compute(&self, data: &[usize]) -> (f64, f64) {
        if lock(&self.failure).is_some() {
            return FAILED;
        }
        let result = Python::attach(|py| {
            self.function
                .call1(py, (data.to_vec(),))
                .and_then(|value| value.extract::<(f64, f64)>(py))
        });
        result.unwrap_or_else(|err| {
            *lock(&self.failure) = Some(err);
            FAILED
        })
    }
}

/// Raises the exception the caller's error function left in `failure`, if any.
pub fn raise_stored(failure: &ErrorSlot) -> pyo3::PyResult<()> {
    match lock(failure).take() {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

/// The slot only ever holds an `Option`, so a panic elsewhere cannot leave
/// it half-written; a poisoned lock is safe to use.
fn lock(slot: &ErrorSlot) -> std::sync::MutexGuard<'_, Option<PyErr>> {
    slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}
