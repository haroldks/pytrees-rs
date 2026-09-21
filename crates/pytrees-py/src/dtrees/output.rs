use dtrees_rs::algorithms::common::types::{SearchStatistics, SearchStrategy};
use dtrees_rs::tree::Tree;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Search output container for decision tree algorithm results.
///
/// This class encapsulates all results and statistics from decision tree
/// construction algorithms, providing a unified interface for accessing
/// tree structures, performance metrics, and detailed search statistics.
///
/// ## Fields
///
/// - `error`: Final classification error of the constructed tree
/// - `tree`: The decision tree structure in JSON format
/// - `statistics`: Detailed search statistics (nodes explored, cache performance, etc.)
/// - `duration`: Total algorithm execution time in seconds
/// - `search`: Search strategy information used during construction
///
/// ## Usage
///
/// This class is typically returned by algorithm functions and should not
/// be instantiated directly by users.
///
/// ```python
/// # Returned by algorithm functions
/// result = classifier.fit(X, y)
/// stats = classifier.stats
///
/// print(f"Error: {stats.error}")
/// print(f"Duration: {stats.duration}s")
/// print(f"Tree: {stats.tree}")
/// print(f"Statistics: {stats.statistics}")
/// ```
#[pyclass(name = "output")]
#[derive(Default, Clone)]
pub struct SearchOutput {
    #[pyo3(get, set)]
    pub(crate) error: f64,
    pub(crate) tree: Tree,
    pub(crate) statistics: SearchStatistics,
    pub(crate) duration: f64,
    pub(crate) search: SearchStrategy,
}

#[pymethods]
impl SearchOutput {
    /// Returns the classification error of the constructed tree.
    ///
    /// # Returns
    ///
    /// The error rate as a float between 0.0 and 1.0, where 0.0 indicates
    /// perfect classification and 1.0 indicates completely incorrect classification.
    #[getter]
    pub fn error(&self) -> PyResult<f64> {
        Ok(self.error)
    }

    /// Returns detailed search statistics as a JSON string.
    ///
    /// The statistics include information about:
    /// - Number of nodes explored during search
    /// - Cache hit/miss ratios
    /// - Memory usage patterns
    /// - Algorithm-specific metrics
    ///
    /// # Returns
    ///
    /// A pretty-printed JSON string containing comprehensive search statistics.
    #[getter]
    pub fn statistics(&self) -> PyResult<String> {
        let json = serde_json::to_string_pretty(&self.statistics).unwrap();
        Ok(json)
    }

    /// The tree as flat arrays, in the layout `ConTreeClassifier.tree_` uses:
    /// node `i` tests `feature[i]` and sends a row left when its value is
    /// below `threshold[i]` (0.5 here, so 0 goes left and 1 right).
    /// `children_left[i] == -1` marks a leaf, which predicts `value[i]`.
    /// Nodes are numbered from the root, depth first; `value` is NaN where
    /// the search set no output, as on a root it found no tree for.
    #[getter]
    pub fn tree_arrays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let mut arrays = TreeArrays::default();
        if !self.tree.is_empty() {
            arrays.push_subtree(&self.tree, self.tree.get_root_index());
        }
        let dict = PyDict::new(py);
        dict.set_item("children_left", PyArray1::from_vec(py, arrays.children_left))?;
        dict.set_item("children_right", PyArray1::from_vec(py, arrays.children_right))?;
        dict.set_item("feature", PyArray1::from_vec(py, arrays.feature))?;
        dict.set_item("threshold", PyArray1::from_vec(py, arrays.threshold))?;
        dict.set_item("value", PyArray1::from_vec(py, arrays.value))?;
        dict.set_item("error", PyArray1::from_vec(py, arrays.error))?;
        Ok(dict)
    }

    /// Returns the total algorithm execution time.
    ///
    /// # Returns
    ///
    /// Duration in seconds as a float.
    #[getter]
    pub fn duration(&self) -> f64 {
        self.duration
    }
}

#[derive(Default)]
struct TreeArrays {
    children_left: Vec<i64>,
    children_right: Vec<i64>,
    feature: Vec<i64>,
    threshold: Vec<f64>,
    value: Vec<f64>,
    error: Vec<f64>,
}

impl TreeArrays {
    /// Appends the subtree at `index`, depth first, and returns the position
    /// its root got. In a dtrees `Tree`, child index 0 means "no child".
    fn push_subtree(&mut self, tree: &Tree, index: usize) -> i64 {
        let details = tree.node_details(index);
        let position = self.feature.len() as i64;
        self.value.push(details.out.unwrap_or(f64::NAN));
        self.error.push(details.error);
        self.children_left.push(-1);
        self.children_right.push(-1);

        let (left, right) = tree.node_children(index);
        match details.test {
            Some(feature) if left != 0 && right != 0 => {
                self.feature.push(feature as i64);
                self.threshold.push(0.5);
                let left = self.push_subtree(tree, left);
                let right = self.push_subtree(tree, right);
                self.children_left[position as usize] = left;
                self.children_right[position as usize] = right;
            }
            _ => {
                self.feature.push(-1);
                self.threshold.push(f64::NAN);
            }
        }
        position
    }
}
