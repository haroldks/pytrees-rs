
use dtrees_rs::algorithms::common::types::{SearchStatistics, SearchStrategy};
use dtrees_rs::algorithms::greedy::{LGDTBuilder, LGDT};
use dtrees_rs::algorithms::optimal::depth2::{ErrorMinimizer, InfoGainMaximizer};
use dtrees_rs::algorithms::common::errors::NativeError;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use dtrees_rs::algorithms::TreeSearchAlgorithm;
use dtrees_rs::cover::Cover;
use crate::common::enums::ExposedSearchStrategy;
use crate::common::types::SearchOutput;

impl ExposedSearchStrategy {
    pub(crate) fn to_lgdt_builder(self, min_sup: usize, max_depth: usize) -> PyResult<Box<dyn LGDTBuilderTrait>> {
        match self {
            ExposedSearchStrategy::LGDTInfoGainMaximizer => {
                Ok(Box::new(
                    LGDTBuilder::<InfoGainMaximizer<NativeError>>::with_default_info_gain_maximizer()
                        .min_support(min_sup)
                        .max_depth(max_depth)
                        .build().map_err(PyValueError::new_err)?
                ))
            },
            ExposedSearchStrategy::LGDTErrorMinimizer => {
                Ok(Box::new(
                    LGDTBuilder::<ErrorMinimizer<NativeError>>::with_default_error_minimizer()
                        .min_support(min_sup)
                        .max_depth(max_depth)
                        .build().map_err(PyValueError::new_err)?
                ))
            },
            _ => {
                Err(PyValueError::new_err(
                    "Search  not supported for LGDT algorithm. Use LGDTInfoGainMaximizer or LGDTErrorMinimizer."
                ))
            },
        }
    }
}

pub trait LGDTBuilderTrait {
    fn fit_and_get_result(&mut self, cover: &mut Cover) -> PyResult<SearchOutput>;

}



impl LGDTBuilderTrait for LGDT<InfoGainMaximizer<NativeError>> {
    fn fit_and_get_result(&mut self, cover: &mut Cover) -> PyResult<SearchOutput> {
        self.fit(cover)
            .map_err(|e| PyValueError::new_err(format!("LGDT fit failed: {:?}", e)))?;

        Ok(SearchOutput {
            error: self.error(),
            tree: self.tree().clone(),
            statistics: SearchStatistics {
                cache_size: 0,
                cache_hits: 0,
                restarts: 0,
                sibling_pruning: 0,
                search_space_size: 0,
                tree_error: self.error(),
                duration: 0.0,
                num_attributes: cover.num_attributes,
                num_samples : cover.num_samples,
            },
            duration: 0.0,
            search: SearchStrategy::LGDTInfoGainMaximizer,
        })
    }
}

impl LGDTBuilderTrait for LGDT<ErrorMinimizer<NativeError>> {
    fn fit_and_get_result(&mut self, cover: &mut Cover) -> PyResult<SearchOutput> {

        self.fit(cover)
            .map_err(|e| PyValueError::new_err(format!("LGDT fit failed: {:?}", e)))?;

        Ok(SearchOutput {
            error: self.error(),
            tree: self.tree().clone(),
            statistics: SearchStatistics {
                cache_size: 0,
                cache_hits: 0,
                restarts: 0,
                sibling_pruning: 0,
                search_space_size: 0,
                tree_error: self.error(),
                duration: 0.0,
                num_attributes: cover.num_attributes,
                num_samples: cover.num_samples,
            },
            duration: 0.0,
            search: SearchStrategy::LGDTErrorMinimizer,
        })
    }
}


