//! Turns numpy arrays into the `Cover` the dtrees searches work on.

use dtrees_rs::bitsets::{BitCollection, Bitset, BitsetInit};
use dtrees_rs::cover::Cover;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

/// One bitset per feature, set where the feature is 1, and one per label.
///
/// `x` must hold only 0 and 1, and `y` labels encoded as `0..k-1`; the
/// Python layer guarantees both, and this checks them again rather than
/// truncating or indexing out of bounds. Without `y`, as for clustering,
/// the cover has no labels.
pub(crate) fn cover(
    x: &PyReadonlyArray2<'_, f64>,
    y: Option<&PyReadonlyArray1<'_, i64>>,
) -> PyResult<Cover> {
    let x = x.as_array();
    let (n_rows, n_features) = x.dim();

    let mut attributes = vec![Bitset::new(BitsetInit::Empty(n_rows)); n_features];
    for ((row, feature), &value) in x.indexed_iter() {
        if value == 1.0 {
            attributes[feature].set(row);
        } else if value != 0.0 {
            return Err(PyValueError::new_err(format!(
                "features must be 0 or 1, found {value} at row {row}, column {feature}"
            )));
        }
    }

    let labels = match y {
        None => vec![],
        Some(y) => {
            let y = y.as_array();
            if y.len() != n_rows {
                return Err(PyValueError::new_err(format!(
                    "X has {n_rows} rows but y has {} labels",
                    y.len()
                )));
            }
            let n_labels = y.iter().max().map_or(0, |&max| max + 1);
            if let Some(&label) = y.iter().find(|&&label| label < 0) {
                return Err(PyValueError::new_err(format!(
                    "labels must be encoded as 0..k-1, found {label}"
                )));
            }
            let mut labels = vec![Bitset::new(BitsetInit::Empty(n_rows)); n_labels as usize];
            for (row, &label) in y.iter().enumerate() {
                labels[label as usize].set(row);
            }
            if let Some(missing) = labels.iter().position(|label| label.count() == 0) {
                return Err(PyValueError::new_err(format!(
                    "labels must be encoded as 0..k-1, but no row has label {missing}"
                )));
            }
            labels
        }
    };

    Ok(Cover::new(attributes, labels, n_rows))
}
