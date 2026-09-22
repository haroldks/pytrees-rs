//! Error functions: the cost of a leaf and what it predicts.

/// The error of a leaf.
pub trait ErrorWrapper: Send + Sync {
    /// `(error, prediction)` of a leaf, given its class counts or its instance
    /// ids, depending on the search's configuration.
    fn compute(&self, data: &[usize]) -> (f64, f64);
}

/// An error function backed by a plain Rust function.
#[derive(Debug, Clone)]
pub struct NativeError {
    function: fn(&[usize]) -> (f64, f64),
}

impl NativeError {
    /// Wraps `function`.
    pub fn new(function: fn(&[usize]) -> (f64, f64)) -> Self {
        NativeError { function }
    }
}

impl Default for NativeError {
    fn default() -> Self {
        Self::new(classification_error)
    }
}

impl ErrorWrapper for NativeError {
    fn compute(&self, data: &[usize]) -> (f64, f64) {
        (self.function)(data)
    }
}

/// Misclassification error of a leaf predicting its majority class, and that
/// class. Ties go to the highest class index. The default error function.
pub fn classification_error(classes_support: &[usize]) -> (f64, f64) {
    let mut max_idx = 0;
    let mut max_value = 0;
    let mut total = 0;
    for (idx, value) in classes_support.iter().enumerate() {
        total += value;
        if *value >= max_value {
            max_value = *value;
            max_idx = idx;
        }
    }
    let error = total - max_value;
    (error as f64, max_idx as f64)
}
