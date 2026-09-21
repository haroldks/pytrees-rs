pub mod types;
pub mod errors;
pub mod enums;

use std::collections::HashSet;
use numpy::PyReadonlyArrayDyn;
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;
use dtrees_rs::bitsets::{BitCollection, Bitset, BitsetInit};
use dtrees_rs::cover::Cover;


pub(crate) fn create_cover_from_numpy(
    input: PyReadonlyArrayDyn<f64>,
    target: Option<&PyReadonlyArrayDyn<f64>>,
) -> PyResult<Cover> {
    let input_array = input.as_array().map(|&x| x as usize );
    let num_samples = input_array.shape()[0];
    let num_features = input_array.shape()[1];

    let mut attributes = Vec::with_capacity(num_features);

    for feature_idx in 0..num_features {
        let mut feature_bitset = Bitset::new(BitsetInit::Empty(num_samples));
        for sample_idx in 0..num_samples {
            let value = input_array[[sample_idx, feature_idx]];
            if value == 1 {
                feature_bitset.set(sample_idx);
            }
        }
        attributes.push(feature_bitset);
    }

    let labels = match target {
        Some(target_array) => {
            let target_array = target_array.as_array().map(|&x| x as usize);

            if target_array.len() != num_samples {
                return Err(PyValueError::new_err(
                    format!("Target length ({}) doesn't match input samples ({})",
                            target_array.len(), num_samples)
                ));
            }

            let mut unique_labels = HashSet::new();
            for sample_idx in 0..num_samples {
                let label = target_array[sample_idx];
                unique_labels.insert(label);
            }

            let num_labels = unique_labels.len();
            let mut labels = vec![Bitset::new(BitsetInit::Empty(num_samples)); num_labels];

            for sample_idx in 0..num_samples {
                let label = target_array[sample_idx];
                labels[label].set(sample_idx);
            }

            labels
        },
        None => vec![],
    };

    Ok(Cover::new(attributes, labels, num_samples))
}
