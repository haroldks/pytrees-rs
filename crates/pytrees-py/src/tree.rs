//! `pytrees._native.tree`: the one routine every estimator's `Tree` uses to
//! send rows down a tree, whichever library grew it.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const LEAF: i64 = -1;

/// The index of the leaf each row of `x` reaches. A row goes left when
/// `x[feature] <= threshold`, as in scikit-learn.
///
/// The arrays are checked before the walk, so a malformed tree is a
/// `ValueError` rather than an out-of-bounds read.
#[pyfunction]
pub fn apply<'py>(
    py: Python<'py>,
    children_left: PyReadonlyArray1<'py, i64>,
    children_right: PyReadonlyArray1<'py, i64>,
    feature: PyReadonlyArray1<'py, i64>,
    threshold: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let (left, right) = (children_left.as_array(), children_right.as_array());
    let (feature, threshold) = (feature.as_array(), threshold.as_array());
    let x = x.as_array();
    let nodes = left.len();
    if [right.len(), feature.len(), threshold.len()] != [nodes; 3] || nodes == 0 {
        return Err(PyValueError::new_err(
            "the tree arrays must all have the same, non-zero length",
        ));
    }
    for node in 0..nodes {
        if left[node] == LEAF {
            continue;
        }
        let child_in_range = |child: i64| child > 0 && (child as usize) < nodes;
        if !child_in_range(left[node]) || !child_in_range(right[node]) {
            return Err(PyValueError::new_err(format!(
                "node {node} has a child outside the tree"
            )));
        }
        if feature[node] < 0 || feature[node] as usize >= x.ncols() {
            return Err(PyValueError::new_err(format!(
                "node {node} tests feature {}, but X has {} columns",
                feature[node],
                x.ncols()
            )));
        }
    }

    let leaves = py.detach(|| {
        x.rows()
            .into_iter()
            .map(|row| {
                let mut node = 0;
                // A path visits each node at most once, so a longer walk
                // means the arrays hold a cycle.
                for _ in 0..nodes {
                    if left[node] == LEAF {
                        return Ok(node as i64);
                    }
                    node = if row[feature[node] as usize] <= threshold[node] {
                        left[node]
                    } else {
                        right[node]
                    } as usize;
                }
                Err(node)
            })
            .collect::<Result<Vec<_>, _>>()
    });
    let leaves = leaves.map_err(|node| {
        PyValueError::new_err(format!("the tree has a cycle through node {node}"))
    })?;
    // Copied into memory numpy owns: an array backed by a Rust Vec cannot
    // have its flags changed, which scipy's sparse indexing does.
    Ok(PyArray1::from_slice(py, &leaves))
}
