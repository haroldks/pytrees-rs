use crate::utils::{
    ExposedBranchingStrategy, ExposedCacheInitStrategy, ExposedDataFormat,
    ExposedDiscrepancyStrategy, ExposedLowerBoundStrategy, ExposedSearchHeuristic,
    ExposedSearchStrategy, ExposedSpecialization, LearningResult, PythonError,
};
use dtrees_rs::cache::Trie;
use dtrees_rs::data::{BinaryData, FileReader};
use dtrees_rs::heuristics::{
    GiniIndex, Heuristic, InformationGain, InformationGainRatio, NoHeuristic,
};
use dtrees_rs::searches::errors::{ErrorWrapper, NativeError};
use dtrees_rs::searches::optimal::{
    Discrepancy, ExponentialDiscrepancy, GenericDL85, LubyDiscrepancy, MonotonicDiscrepancy,
    TopKDL85, LDSDL85,
};
use dtrees_rs::searches::{
    BranchingStrategy, CacheInitStrategy, LowerBoundStrategy, NodeExposedData, SearchStrategy,
    Specialization, Statistics,
};
use dtrees_rs::structures::RevBitset;
use numpy::PyReadonlyArrayDyn;
use pyo3::{pyclass, pymethods, PyObject};
// TODO : merge with LDS
#[pyclass]
pub struct PyTopKDl85 {
    learner: TopKDL85<Trie, dyn Discrepancy + Send, dyn ErrorWrapper + Send, dyn Heuristic + Send>,
    pub(crate) error: f64,
    pub(crate) results: LearningResult,
    pub(crate) duration: f64,
    structure: RevBitset,
}

#[pymethods]
impl PyTopKDl85 {
    #[new]
    #[pyo3(signature = (min_sup=1, max_depth=2, init_k=1, budget=<usize>::MAX, error=<f64>::INFINITY, time=600, one_time_sort=true, cache_init_size=0, discrepancy_strategy=ExposedDiscrepancyStrategy::Monotonic, exposed_data_format=ExposedDataFormat::ClassSupports, specialization=ExposedSpecialization::Murtree, lower_bound=ExposedLowerBoundStrategy::Similarity, branching_type=ExposedBranchingStrategy::Dynamic, heuristic=ExposedSearchHeuristic::None_, cache_init_strategy=ExposedCacheInitStrategy::None_, error_function=None))]
    pub fn new(
        min_sup: usize,
        max_depth: usize,
        init_k: usize,
        budget: usize,
        error: f64,
        time: usize,
        one_time_sort: bool,
        cache_init_size: usize,
        discrepancy_strategy: ExposedDiscrepancyStrategy,
        exposed_data_format: ExposedDataFormat,
        specialization: ExposedSpecialization,
        lower_bound: ExposedLowerBoundStrategy,
        branching_type: ExposedBranchingStrategy,
        heuristic: ExposedSearchHeuristic,
        cache_init_strategy: ExposedCacheInitStrategy,
        error_function: Option<PyObject>,
    ) -> Self {
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

        let heuristic: Box<dyn Heuristic + Send> = match heuristic {
            ExposedSearchHeuristic::InformationGain => Box::<InformationGain>::default(),
            ExposedSearchHeuristic::InformationGainRatio => Box::<InformationGainRatio>::default(),
            ExposedSearchHeuristic::GiniIndex => Box::<GiniIndex>::default(),
            ExposedSearchHeuristic::None_ => Box::<NoHeuristic>::default(),
        };

        let discrepancy_function: Box<dyn Discrepancy + Send> = match discrepancy_strategy {
            ExposedDiscrepancyStrategy::Monotonic => Box::<MonotonicDiscrepancy>::default(),
            ExposedDiscrepancyStrategy::Exponential => Box::<ExponentialDiscrepancy>::default(),
            ExposedDiscrepancyStrategy::Luby => Box::<LubyDiscrepancy>::default(),
        };

        let external_error: Box<dyn ErrorWrapper + Send> = match error_function {
            Some(function) => {
                specialization = Specialization::None_;
                Box::new(PythonError::new(function))
            }
            None => Box::<NativeError>::default(),
        };

        // TODO : Allow multiple caching strategy
        let cache = Box::<Trie>::default();

        let mut learner = TopKDL85::new(
            min_sup,
            max_depth,
            init_k,
            budget,
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
            discrepancy_function,
            external_error,
            heuristic,
        );

        let search_strategy = match discrepancy_strategy {
            ExposedDiscrepancyStrategy::Monotonic => SearchStrategy::TopKSearchMonotonic,
            ExposedDiscrepancyStrategy::Exponential => SearchStrategy::TopKSearchExponential,
            ExposedDiscrepancyStrategy::Luby => SearchStrategy::TopKSearchLuby,
        };
        learner.statistics.constraints.search_strategy = search_strategy;
        let statistics = learner.statistics;
        Self {
            learner,
            error: <f64>::INFINITY,
            results: LearningResult {
                error,
                tree: Default::default(),
                constraints: statistics.constraints,
                statistics,
                duration: 0.0,
            },
            duration: 0.0,
            structure: Default::default(),
        }
    }

    pub fn fit(
        &mut self,
        input: PyReadonlyArrayDyn<f64>,
        target: Option<PyReadonlyArrayDyn<f64>>,
    ) -> LearningResult {
        if target.is_none() {
            if let NodeExposedData::ClassesSupport = self.results.constraints.node_exposed_data {
                panic!("When target (y) is not specified cover (with tids) must be used for error computation")
            }
        }

        // Objects initialization start
        let input = input.as_array().map(|a| *a as usize);
        let target = match target.is_some() {
            true => Some(target.unwrap().as_array().map(|a| *a as usize)),
            false => None,
        };
        let dataset = BinaryData::read_from_numpy(&input, target.as_ref());
        let mut structure = RevBitset::new(&dataset);

        self.learner.fit(&mut structure);

        let statistics = self.learner.statistics;
        let results = LearningResult {
            error: statistics.tree_error,
            tree: self.learner.tree.clone(),
            constraints: statistics.constraints,
            statistics,
            duration: statistics.duration.as_secs_f64(),
        };
        results
    }

    /// FIXME : Partial fit restart and lower bound not working quite well
    #[pyo3(signature = (input, target=None, first=true, runtime=10))]
    pub fn partial_fit(
        &mut self,
        input: PyReadonlyArrayDyn<f64>,
        target: Option<PyReadonlyArrayDyn<f64>>,
        first: bool,
        runtime: usize,
    ) -> LearningResult {
        if target.is_none() {
            if let NodeExposedData::ClassesSupport = self.results.constraints.node_exposed_data {
                panic!("When target (y) is not specified cover (with tids) must be used for error computation")
            }
        }

        // Objects initialization start
        if first {
            let input = input.as_array().map(|a| *a as usize);
            let target = match target.is_some() {
                true => Some(target.unwrap().as_array().map(|a| *a as usize)),
                false => None,
            };
            let dataset = BinaryData::read_from_numpy(&input, target.as_ref());
            self.structure = RevBitset::new(&dataset);
        }

        self.learner.partial_fit(&mut self.structure, Some(runtime));

        let statistics = self.learner.statistics;
        let results = LearningResult {
            error: statistics.tree_error,
            tree: self.learner.tree.clone(),
            constraints: statistics.constraints,
            statistics,
            duration: statistics.duration.as_secs_f64(),
        };
        results
    }

    pub fn is_optimal(&self) -> bool {
        self.learner.is_optimal()
    }
}
