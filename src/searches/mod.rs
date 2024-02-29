pub mod errors;
pub mod greedy;
pub mod optimal;
mod utils;

pub use utils::*;

fn deduce_sibling_error(parent_supports: &[usize], child_supports: &[usize]) -> Vec<usize> {
    parent_supports
        .iter()
        .zip(child_supports.iter())
        .map(|(root, child)| *root - *child)
        .collect()
}
