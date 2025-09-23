use crate::common::create_cover_from_numpy;

use dtrees_rs::algorithms::common::errors::{ErrorWrapper, NativeError};
use dtrees_rs::algorithms::common::heuristics::{GiniIndex, Heuristic, InformationGain, NoHeuristic, WeightedEntropy};
use dtrees_rs::algorithms::optimal::depth2::ErrorMinimizer;
use dtrees_rs::algorithms::optimal::dl85::{DL85Builder, DL85};
use dtrees_rs::algorithms::optimal::rules::{DiscrepancyRule, GainRule, PurityRule, TopkRule, RuleManager, common::TimeLimitRule, Rule};
use dtrees_rs::algorithms::TreeSearchAlgorithm;
use dtrees_rs::caching::Trie;
use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use dtrees_rs::algorithms::optimal::dl85::config::DL85Config;
use dtrees_rs::cover::Cover;
use log::error;
use crate::common::enums::{ExposedBranchingPolicy, ExposedDepth2Policy, ExposedHeuristic, ExposedLowerBoundPolicy, ExposedNodeDataType,};
use crate::common::errors::PythonError;
use crate::common::types::{ExposedDiscrepancyRule, ExposedGainRule, ExposedPurityRule, ExposedRestartRule, ExposedTopKRule, SearchOutput};



#[pyclass]
pub struct PyDL85 {
    learner: DL85<Trie, ErrorMinimizer<dyn ErrorWrapper>, dyn ErrorWrapper, dyn Heuristic>,
    config: DL85Config,
    cover: Cover,
    statistics: SearchOutput,
    has_data: bool,
}


#[pymethods]
impl PyDL85 {

    #[new]
    #[pyo3(signature = (
        min_sup=1,
        max_depth=2,
        time_limit=600.0,
        always_sort=true,
        heuristic=ExposedHeuristic::Disabled,
        depth2_policy=ExposedDepth2Policy::Disabled,
        lower_bound=ExposedLowerBoundPolicy::Disabled,
        branching_policy=ExposedBranchingPolicy::Default,
        data_type=ExposedNodeDataType::ClassSupports,
        discrepancy=None,
        gain=None,
        topk=None,
        restart=None,
        purity=None,
        error_function=None,
    ))]
    pub fn new(
        min_sup: usize,
        max_depth: usize,
        time_limit: f64,
        always_sort: bool,
        heuristic: ExposedHeuristic,
        depth2_policy: ExposedDepth2Policy,
        lower_bound: ExposedLowerBoundPolicy,
        branching_policy: ExposedBranchingPolicy,
        data_type: ExposedNodeDataType,
        // Rule parameters
        discrepancy: Option<ExposedDiscrepancyRule>,
        gain: Option<ExposedGainRule>,
        topk: Option<ExposedTopKRule>,
        restart: Option<ExposedRestartRule>,
        purity: Option<ExposedPurityRule>,
        error_function: Option<Py<PyAny>>,
    ) -> PyResult<Self> {

        let heuristic_fn: Box<dyn Heuristic> = heuristic.into();
        let depth2_policy = depth2_policy.into();
        let lower_bound_policy = lower_bound.into();
        let branching_policy = branching_policy.into();
        let data_type = data_type.into();


        // let error_fn_copy = error_function.

        let error_fn: Box<dyn ErrorWrapper> = match &error_function {
            None => Box::new(NativeError::default()),
            Some(function) => {
                let mut error: Box<dyn ErrorWrapper> = Box::<NativeError>::default();
                Python::attach(|py| {
                    error = Box::new(PythonError::new(function.clone_ref(py)))
                });
                error
            } ,
        };




        let depth2_search: Box<ErrorMinimizer<dyn ErrorWrapper>> = match &error_function {
            None => Box::new(ErrorMinimizer::new(Box::<NativeError>::default())),
            Some(function) => {
                let mut d2: Box<ErrorMinimizer<dyn ErrorWrapper>> = Box::new(ErrorMinimizer::new(Box::<NativeError>::default()));
                Python::attach(|py| {
                    d2 = Box::new(ErrorMinimizer::new(Box::new(PythonError::new(function.clone_ref(py)))))
                });
                d2
            } ,
        };

            Box::new(ErrorMinimizer::new(Box::<NativeError>::default()));

        // Configure cache
        let cache = Box::new(Trie::default());

        // Configure rules
        let mut node_rules: Vec<Box<dyn Rule>> = vec![];
        let mut search_rules: Vec<Box<dyn Rule>> = vec![];

        // Add discrepancy rule if specified
        if let Some(rule) = discrepancy {
            let discrepancy_rule = DiscrepancyRule::from(rule);
            search_rules.push(Box::new(discrepancy_rule));
        }

        // Add gain rule if specified
        if let Some(rule) = gain {
            let gain_rule = GainRule::from(rule);
            search_rules.push(Box::new(gain_rule));
        }

        // Add purity rule if specified
        if let Some(rule) = purity {
            let purity_rule = PurityRule::from(rule);
            node_rules.push(Box::new(purity_rule));
        }

        // Add topk rule if specified
        if let Some(rule) = topk {
            let topk_rule = TopkRule::from(rule);
            search_rules.push(Box::new(topk_rule));
        }

        // Create restart time limit rule
        if let Some(rule) = restart {
            let restart_rule = TimeLimitRule::from(rule);
            search_rules.push(Box::new(restart_rule));
        }

        // Build DL85 algorithm using the builder pattern
        let mut learner = DL85Builder::default()
            .max_depth(max_depth)
            .min_support(min_sup)
            .max_time(time_limit)
            .specialization(depth2_policy)
            .always_sort(always_sort)
            .branching_strategy(branching_policy)
            .lower_bound_strategy(lower_bound_policy)
            .node_exposed_data(data_type)
            .cache(cache)
            .heuristic(heuristic_fn)
            .depth2_search(depth2_search)
            .add_search_rules(search_rules)
            .add_node_rules(node_rules)
            .error_function(error_fn)
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to build DL85: {:?}", e)))?;

        let config = learner.config();

        Ok( Self {
            learner,
            config,
            cover: Cover::new(vec![], vec![], 0),
            statistics: SearchOutput::default(),
            has_data: false
        }
        )

    }


    pub fn load_data(&mut self, input: PyReadonlyArrayDyn<f64>, target: Option<PyReadonlyArrayDyn<f64>>) -> PyResult<()>  {
        let cover = create_cover_from_numpy(input, target.as_ref())?;
        self.cover = cover;
        self.has_data = true;
        Ok(())
    }

    pub fn partial_fit(&mut self) -> PyResult<()>{
        if self.has_data {
            return Err(PyValueError::new_err("Load data before using partial fit or use fit directly."))
        }

        self.learner.partial_fit(&mut self.cover);
        self.update_stats();
        Ok(())
    }


    pub fn fit(&mut self, input: PyReadonlyArrayDyn<f64>, target: Option<PyReadonlyArrayDyn<f64>>) -> PyResult<()>{
        self.load_data(input, target);
        self.learner.fit(&mut self.cover).map_err(|x| PyValueError::new_err(format!("Failed to fit due to {:?}", x)))?;
        self.update_stats();
        Ok(())

    }


    #[getter]
    pub fn stats(&self) -> PyResult<SearchOutput> {
        Ok(self.statistics.clone())
    }

    fn update_stats(&mut self) {
        self.statistics.error = self.learner.error();
        self.statistics.duration = self.learner.elapsed_seconds();
        self.statistics.statistics = *self.learner.statistics();
        self.statistics.tree = self.learner.tree().clone();
    }

}
