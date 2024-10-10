pub mod partial;

use crate::utils::{
    ExposedBranchingStrategy, ExposedCacheInitStrategy, ExposedDataFormat,
    ExposedLowerBoundStrategy, ExposedSearchHeuristic, ExposedSpecialization, LearningResult,
    PythonError,
};
use dtrees_rs::cache::trie::Trie;
use dtrees_rs::data::{BinaryData, FileReader};
use dtrees_rs::heuristics::{
    GiniIndex, Heuristic, InformationGain, InformationGainRatio, NoHeuristic,
};
use dtrees_rs::searches::errors::{ErrorWrapper, NativeError};
use dtrees_rs::searches::optimal::DL85;
use dtrees_rs::searches::{
    BranchingStrategy, CacheInitStrategy, LowerBoundStrategy, NodeExposedData, Specialization,
};
use dtrees_rs::structures::RevBitset;
use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::*;

pub use partial::PyGenericDl85;
#[pyfunction]
#[pyo3(name = "dl85")]
#[pyo3(signature = (input, target=None, min_sup=1, max_depth=2, time=600, cache_init_size=0, error=<f64>::INFINITY, one_time_sort=true, exposed_data_format=ExposedDataFormat::ClassSupports, specialization=ExposedSpecialization::Murtree, lower_bound=ExposedLowerBoundStrategy::Similarity, branching_type=ExposedBranchingStrategy::Dynamic, heuristic=ExposedSearchHeuristic::None_, cache_init_strategy=ExposedCacheInitStrategy::None_, error_function=None,))]
pub(crate) fn optimal_search_dl85(
    input: PyReadonlyArrayDyn<f64>,
    target: Option<PyReadonlyArrayDyn<f64>>,
    min_sup: usize,
    max_depth: usize,
    time: usize,
    cache_init_size: usize,
    error: f64,
    one_time_sort: bool,
    exposed_data_format: ExposedDataFormat,
    specialization: ExposedSpecialization,
    lower_bound: ExposedLowerBoundStrategy,
    branching_type: ExposedBranchingStrategy,
    heuristic: ExposedSearchHeuristic,
    cache_init_strategy: ExposedCacheInitStrategy,
    error_function: Option<PyObject>,
) -> LearningResult {
    if target.is_none() {
        if let ExposedDataFormat::ClassSupports = exposed_data_format {
            panic!("When target (y) is not specified cover (with tids) must be used for error computation")
        }
    }

    let data_format = match exposed_data_format {
        ExposedDataFormat::Tids => NodeExposedData::Tids,
        ExposedDataFormat::ClassSupports => NodeExposedData::ClassesSupport,
    };

    let cache_init_strategy = match cache_init_strategy {
        ExposedCacheInitStrategy::DynamicAllocation => CacheInitStrategy::DynamicAllocation,
        ExposedCacheInitStrategy::UserAllocation => CacheInitStrategy::UserAllocation,
        ExposedCacheInitStrategy::None_ => CacheInitStrategy::None_,
    };

    let mut specialization = match specialization {
        ExposedSpecialization::Murtree => Specialization::Murtree,
        ExposedSpecialization::None_ => Specialization::None_,
    };

    let lower_bound_strategy = match lower_bound {
        ExposedLowerBoundStrategy::Similarity => LowerBoundStrategy::Similarity,
        ExposedLowerBoundStrategy::None_ => LowerBoundStrategy::None_,
    };

    let branching_strategy = match branching_type {
        ExposedBranchingStrategy::Dynamic => BranchingStrategy::Dynamic,
        ExposedBranchingStrategy::None_ => BranchingStrategy::None_,
    };

    let heuristic: Box<dyn Heuristic> = match heuristic {
        ExposedSearchHeuristic::InformationGain => Box::<InformationGain>::default(),
        ExposedSearchHeuristic::InformationGainRatio => Box::<InformationGainRatio>::default(),
        ExposedSearchHeuristic::GiniIndex => Box::<GiniIndex>::default(),
        ExposedSearchHeuristic::None_ => Box::<NoHeuristic>::default(),
    };

    // Objects initialization start
    let input = input.as_array().map(|a| *a as usize);
    let target = match target.is_some() {
        true => Some(target.unwrap().as_array().map(|a| *a as usize)),
        false => None,
    };
    let dataset = BinaryData::read_from_numpy(&input, target.as_ref());
    let mut structure = RevBitset::new(&dataset);

    let external_error: Box<dyn ErrorWrapper> = match error_function {
        Some(function) => {
            specialization = Specialization::None_;
            Box::new(PythonError::new(function))
        }
        None => Box::<NativeError>::default(),
    };

    // TODO : Allow multiple caching strategy
    let cache = Box::<Trie>::default();

    let mut learner = DL85::new(
        min_sup,
        max_depth,
        error,
        time,
        one_time_sort,
        cache_init_size,
        cache_init_strategy,
        specialization,
        lower_bound_strategy,
        branching_strategy,
        data_format,
        cache,
        external_error,
        heuristic,
    );

    learner.fit(&mut structure);

    LearningResult {
        error: learner.statistics.tree_error,
        tree: learner.tree,
        constraints: learner.statistics.constraints,
        statistics: learner.statistics,
        duration: learner.statistics.duration.as_secs_f64(),
    }
}
