pub trait ErrorWrapper {
    fn compute(&self, data: &[usize]) -> (f64, f64);
}

#[derive(Debug, Clone)]
pub struct NativeError {
    function: fn(&[usize]) -> (f64, f64),
}

impl NativeError {
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

pub fn classification_error(classes_support: &[usize]) -> (f64, f64) {
    // TODO: Move it out of this impl
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
