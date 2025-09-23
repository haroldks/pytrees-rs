use pyo3::{pyclass, pymethods, PyResult};
use dtrees_rs::algorithms::common::types::{SearchStatistics, SearchStrategy};
use dtrees_rs::algorithms::optimal::rules::{DiscrepancyRule, Exponential, GainRule, Luby, Monotonic, PurityRule, StepStrategy, TopkRule};
use dtrees_rs::tree::Tree;
use crate::common::enums::ExposedStepStrategy;
use pyo3::prelude::*;
use dtrees_rs::algorithms::optimal::rules::common::TimeLimitRule;

#[pyclass(name = "output")]
#[derive(Default, Clone)]
pub struct SearchOutput {
    #[pyo3(get, set)]
    pub(crate) error: f64,
    pub(crate) tree: Tree,
    pub(crate) statistics: SearchStatistics,
    pub(crate) duration: f64,
    pub(crate) search: SearchStrategy
}




#[pymethods]
impl SearchOutput {
    // Could be done with paste!

    #[getter]
    pub fn statistics(&self) -> PyResult<String> {
        let json = serde_json::to_string_pretty(&self.statistics).unwrap();
        Ok(json)
    }


    #[getter]
    pub fn tree(&self) -> PyResult<String> {
        let json = serde_json::to_string_pretty(&self.tree).unwrap();
        Ok(json)
    }

    #[getter]
    pub fn duration(&self) -> f64 {
        self.duration
    }
}

#[pyclass]
#[derive(Copy, Clone)]
pub struct ExposedDiscrepancyRule {
    initial_value: usize,
    limit: usize,
    step_strategy: ExposedStepStrategy,
    base: usize
}

#[pymethods]
impl ExposedDiscrepancyRule {
    #[new]
    #[pyo3(signature = (initial_value=0, limit=usize::MAX, step_strategy=ExposedStepStrategy::Monotonic, base=1))]
    pub fn new(initial_value: usize, limit:usize, step_strategy:ExposedStepStrategy, base: usize) -> Self {
        Self {
            initial_value,
            limit,
            step_strategy,
            base,
        }
    }

}


impl From<ExposedDiscrepancyRule> for DiscrepancyRule {
    fn from(value: ExposedDiscrepancyRule) -> Self {

        let step: Box<dyn StepStrategy> = match value.step_strategy {
            ExposedStepStrategy::Monotonic => Box::new(Monotonic::new(value.base)),
            ExposedStepStrategy::Exponential => Box::new(Exponential::new(value.base)),
            ExposedStepStrategy::Luby => Box::new(Luby::new(value.base))
        };
        DiscrepancyRule::new(value.limit, step).with_budget(value.initial_value)
    }
}


#[pyclass]
#[derive(Copy, Clone)]
pub struct ExposedGainRule {
    min_gain: f64,
    epsilon:f64,
    limit: f64,
    step_strategy: ExposedStepStrategy,
    base: usize
}

#[pymethods]
impl ExposedGainRule {
    #[new]
    #[pyo3(signature = (min_gain=0.0, epsilon=1e-4, limit=6.0, step_strategy=ExposedStepStrategy::Monotonic, base=1))]
    pub fn new(min_gain: f64, epsilon: f64, limit: f64,  step_strategy:ExposedStepStrategy, base: usize) -> Self {
        Self {
            min_gain,
            epsilon,
            limit,
            step_strategy,
            base
        }
    }
}


impl From<ExposedGainRule> for GainRule {
    fn from(value: ExposedGainRule) -> Self {
        let step: Box<dyn StepStrategy> = match value.step_strategy {
            ExposedStepStrategy::Monotonic => Box::new(Monotonic::new(value.base)),
            ExposedStepStrategy::Exponential => Box::new(Exponential::new(value.base)),
            ExposedStepStrategy::Luby => Box::new(Luby::new(value.base))
        };
        GainRule::new(value.min_gain, value.epsilon, value.limit, step)
    }
}

#[pyclass]
#[derive(Copy, Clone)]
pub struct ExposedPurityRule {
    min_purity: f64,
    epsilon: f64
}

#[pymethods]
impl ExposedPurityRule {

    #[new]
    #[pyo3(signature = (min_purity=0.0, epsilon=1e-4))]
    pub fn new(min_purity: f64, epsilon: f64) -> Self {
        Self {
            min_purity,
            epsilon
        }
    }

}

impl From<ExposedPurityRule> for PurityRule {
    fn from(value: ExposedPurityRule) -> Self {
        PurityRule::new(value.min_purity, value.epsilon)
    }
}


#[pyclass]
#[derive(Copy, Clone)]
pub struct ExposedTopKRule {
    initial_value: usize,
    limit: usize,
    step_strategy: ExposedStepStrategy,
    base: usize
}

#[pymethods]
impl ExposedTopKRule {
    #[new]
    #[pyo3(signature = (initial_value=0, limit=usize::MAX, step_strategy=ExposedStepStrategy::Monotonic, base=1))]
    pub fn new(initial_value: usize, limit:usize, step_strategy:ExposedStepStrategy, base: usize) -> Self {
        Self {
            initial_value,
            limit,
            step_strategy,
            base,
        }
    }

}


impl From<ExposedTopKRule> for TopkRule {
    fn from(value: ExposedTopKRule) -> Self {

        let step: Box<dyn StepStrategy> = match value.step_strategy {
            ExposedStepStrategy::Monotonic => Box::new(Monotonic::new(value.base)),
            ExposedStepStrategy::Exponential => Box::new(Exponential::new(value.base)),
            ExposedStepStrategy::Luby => Box::new(Luby::new(value.base))
        };
        TopkRule::new(value.limit, step).with_budget(value.initial_value)
    }
}


#[pyclass]
#[derive(Copy, Clone)]
pub struct ExposedRestartRule {
    limit: f64
}

#[pymethods]
impl ExposedRestartRule {
    #[new]
    #[pyo3(signature = (limit=1.0))]
    pub fn new(limit: f64) -> Self {
        Self {
            limit
        }
    }

}

impl From<ExposedRestartRule> for TimeLimitRule {
    fn from(value: ExposedRestartRule) -> Self {
        TimeLimitRule::new(value.limit).relaxable()
    }
}
