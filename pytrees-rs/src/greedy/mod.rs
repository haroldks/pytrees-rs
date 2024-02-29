use crate::utils::{ExposedSearchStrategy, LearningResult};
use dtrees_rs::data::{BinaryData, FileReader};
use dtrees_rs::searches::greedy::LGDT;
use dtrees_rs::searches::SearchStrategy;
use dtrees_rs::structures::RevBitset;
use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "lgdt")]
pub(crate) fn search_lgdt(
    input: PyReadonlyArrayDyn<f64>,
    target: PyReadonlyArrayDyn<f64>,
    search_strategy: ExposedSearchStrategy,
    min_sup: usize,
    max_depth: usize,
) -> LearningResult {
    let search_strategy = match search_strategy {
        ExposedSearchStrategy::LessGreedyInfoGain => SearchStrategy::LessGreedyInfoGain,
        ExposedSearchStrategy::LessGreedyMurtree => SearchStrategy::LessGreedyMurtree,
        _ => panic!("Invalid strategy for this approach"),
    };

    let input = input.as_array().map(|a| *a as usize);
    let target = target.as_array().map(|a| *a as usize);
    let dataset = BinaryData::read_from_numpy(&input, Some(&target));
    let mut structure = RevBitset::new(&dataset);

    let mut learner = LGDT::new(min_sup, max_depth, search_strategy);

    learner.fit(&mut structure);

    LearningResult {
        error: learner.error,
        tree: learner.tree.clone(),
        constraints: learner.constraints,
        statistics: learner.statistics,
    }
}
