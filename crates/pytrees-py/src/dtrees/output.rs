//! What a fitted dtrees search hands back to Python.

use dtrees_rs::algorithms::common::types::SearchStatistics;
use dtrees_rs::tree::Tree;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The tree as flat arrays, in the layout `RawConTree.tree_arrays` uses:
/// node `i` tests `feature[i]` and sends a row left when its value is below
/// `threshold[i]` (0.5 here, so 0 goes left and 1 right).
/// `children_left[i] == -1` marks a leaf, which predicts `value[i]`. Nodes
/// are numbered from the root, depth first; `value` is NaN where the search
/// set no output, as on a root it found no tree for.
pub(crate) fn tree_arrays<'py>(py: Python<'py>, tree: &Tree) -> PyResult<Bound<'py, PyDict>> {
    let mut arrays = TreeArrays::default();
    if !tree.is_empty() {
        arrays.push_subtree(tree, tree.get_root_index());
    }
    let dict = PyDict::new(py);
    dict.set_item(
        "children_left",
        PyArray1::from_vec(py, arrays.children_left),
    )?;
    dict.set_item(
        "children_right",
        PyArray1::from_vec(py, arrays.children_right),
    )?;
    dict.set_item("feature", PyArray1::from_vec(py, arrays.feature))?;
    dict.set_item("threshold", PyArray1::from_vec(py, arrays.threshold))?;
    dict.set_item("value", PyArray1::from_vec(py, arrays.value))?;
    dict.set_item("error", PyArray1::from_vec(py, arrays.error))?;
    Ok(dict)
}

/// The search counters, under the names `RawConTree.statistics` uses where
/// the two libraries count the same thing.
pub(crate) fn statistics<'py>(
    py: Python<'py>,
    stats: &SearchStatistics,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("error", stats.tree_error)?;
    dict.set_item("duration", stats.duration)?;
    dict.set_item("cache_size", stats.cache_size)?;
    dict.set_item("cache_hits", stats.cache_hits)?;
    dict.set_item("restarts", stats.restarts)?;
    dict.set_item("sibling_pruning", stats.sibling_pruning)?;
    dict.set_item("search_space_size", stats.search_space_size)?;
    dict.set_item("n_samples", stats.num_samples)?;
    dict.set_item("n_features", stats.num_attributes)?;
    Ok(dict)
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
