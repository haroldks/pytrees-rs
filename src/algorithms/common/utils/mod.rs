pub mod heuristics;

use crate::cover::Cover;
use crate::globals::item;

pub fn find_valid_split_attributes(cover: &mut Cover, min_sup: usize, candidates: Option<&[usize]>, previous: Option<usize>) -> Vec<usize> {
    match candidates {
        Some(attrs) => {
            let mut valid = Vec::new();
            for &attribute in attrs {
                if previous.is_some() && previous.unwrap() == attribute {
                    continue;
                }

                let left_count = cover.count_if_branch_on(item(attribute, 0));
                let right_count = cover.count_if_branch_on(item(attribute, 1));

                if left_count >= min_sup && right_count >= min_sup {
                    valid.push(attribute);
                }
            }
            valid
        },

        None => {
            let num_attributes = cover.num_attributes;
            let mut valid_attributes = Vec::with_capacity(num_attributes);

            for attr_idx in 0..num_attributes {
                if previous.is_some() && previous.unwrap() == attr_idx {
                    continue;
                }

                let left_count = cover.count_if_branch_on(item(attr_idx, 0));
                let right_count = cover.count_if_branch_on(item(attr_idx, 1));

                if left_count >= min_sup && right_count >= min_sup {
                    valid_attributes.push(attr_idx);
                }
            }

            valid_attributes
        }
    }
}

pub fn build_labels_count_distribution_matrix(
    cover: &mut Cover,
    candidates: &[usize],
) -> Vec<Vec<Vec<usize>>> {
    let size = candidates.len();
    let mut matrix = vec![vec![vec![]; size]; size];

    for i in 0..size {
        cover.branch_on(item(candidates[i], 1));

        let first_split_distribution = cover.labels_count();
        matrix[i][i] = first_split_distribution;

        for j in i + 1..size {
            cover.branch_on(item(candidates[j], 1));
            let second_split_distribution = cover.labels_count();
            matrix[i][j] = second_split_distribution.clone();
            matrix[j][i] = second_split_distribution;
            cover.backtrack();
        }
        cover.backtrack();
    }
    matrix
}

#[inline]
pub fn deduce_sibling_error(parent_supports: &[usize], child_supports: &[usize]) -> Vec<usize> {
    parent_supports
        .iter()
        .zip(child_supports.iter())
        .map(|(root, child)| *root - *child)
        .collect()
}

#[inline]
pub fn deduce_sibling_error_with_buffer(
    parent: &[usize],
    sibling: &[usize],
    buffer: &mut [usize]
) {
    for i in 0..parent.len() {
        buffer[i] = parent[i] - sibling[i];
    }
}