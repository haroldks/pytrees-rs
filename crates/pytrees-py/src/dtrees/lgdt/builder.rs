use crate::dtrees::options::LgdtCriterion;
use crate::dtrees::output::SearchOutput;
use dtrees_rs::algorithms::common::errors::NativeError;
use dtrees_rs::algorithms::common::types::{SearchStatistics, SearchStrategy};
use dtrees_rs::algorithms::greedy::{LGDTBuilder, LGDT};
use dtrees_rs::algorithms::optimal::depth2::{ErrorMinimizer, InfoGainMaximizer};
use dtrees_rs::algorithms::TreeSearchAlgorithm;
use dtrees_rs::cover::Cover;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The LGDT search for `criterion`, configured and ready to fit.
pub(crate) fn lgdt_learner(
    criterion: LgdtCriterion,
    min_sup: usize,
    max_depth: usize,
) -> PyResult<Box<dyn LGDTBuilderTrait>> {
    Ok(match criterion {
        LgdtCriterion::InformationGain => Box::new(
            LGDTBuilder::<InfoGainMaximizer<NativeError>>::with_default_info_gain_maximizer()
                .min_support(min_sup)
                .max_depth(max_depth)
                .build()
                .map_err(PyValueError::new_err)?,
        ),
        LgdtCriterion::Error => Box::new(
            LGDTBuilder::<ErrorMinimizer<NativeError>>::with_default_error_minimizer()
                .min_support(min_sup)
                .max_depth(max_depth)
                .build()
                .map_err(PyValueError::new_err)?,
        ),
    })
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
                num_samples: cover.num_samples,
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
