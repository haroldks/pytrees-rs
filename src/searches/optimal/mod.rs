pub mod d2;

mod dl85;

use crate::cache::Caching;
use crate::heuristics::Heuristic;
use crate::searches::errors::ErrorWrapper;
use crate::searches::{
    BranchingStrategy, CacheInitStrategy, LowerBoundStrategy, NodeExposedData, SearchStrategy,
    Specialization, Statistics,
};
use crate::structures::Structure;
use crate::tree::Tree;
pub use d2::Depth2Algorithm;
pub use dl85::discrepancies::*;
pub use dl85::dyntopk::DynTopkDL85;
pub use dl85::lds::LDSDL85;
pub use dl85::purity::PurityDL85;
pub use dl85::relative_gain::RelativeGainDL85;
pub use dl85::restart::RestartDL85;
pub use dl85::topk::TopKDL85;
pub use dl85::DL85;

// TODO : Add support to discrepancy search
pub enum GenericDL85<C, E, H>
where
    C: Caching + ?Sized,
    E: ErrorWrapper + ?Sized + Send,
    H: Heuristic + ?Sized + Send,
{
    BaseDL85(DL85<C, E, H>),
    RestartDL85(RestartDL85<C, E, H>),
    PurityDL85(PurityDL85<C, E, H>),
}

impl<C, E, H> GenericDL85<C, E, H>
where
    C: Caching + ?Sized,
    E: ErrorWrapper + ?Sized + Send,
    H: Heuristic + ?Sized + Send,
{
    pub fn new(
        min_sup: usize,
        max_depth: usize,
        max_error: f64,
        restart_time: Option<usize>,
        max_time: usize,
        purity: Option<f64>,
        epsilon: Option<f64>,
        one_time_sort: bool,
        cache_init_size: usize,
        cache_init_strategy: CacheInitStrategy,
        specialization: Specialization,
        lower_bound_strategy: LowerBoundStrategy,
        branching: BranchingStrategy,
        data_format: NodeExposedData,
        cache: Box<C>,
        error_function: Box<E>,
        heuristic: Box<H>,
        fit_type: SearchStrategy,
    ) -> GenericDL85<C, E, H> {
        match fit_type {
            SearchStrategy::RestartTimeout => {
                assert!(restart_time.is_some(), "Provide a restart time");
                let learner = RestartDL85::new(
                    min_sup,
                    max_depth,
                    max_error,
                    restart_time,
                    max_time,
                    one_time_sort,
                    cache_init_size,
                    cache_init_strategy,
                    specialization,
                    lower_bound_strategy,
                    branching,
                    data_format,
                    cache,
                    error_function,
                    heuristic,
                );
                GenericDL85::RestartDL85(learner)
            }
            SearchStrategy::PurityLimit => {
                assert!(
                    purity.is_some() && epsilon.is_some(),
                    "You must specify the initial purity threshold and the increasing ratio"
                );
                let learner = PurityDL85::new(
                    min_sup,
                    max_depth,
                    max_error,
                    purity.unwrap(),
                    epsilon.unwrap(),
                    max_time,
                    one_time_sort,
                    cache_init_size,
                    cache_init_strategy,
                    specialization,
                    lower_bound_strategy,
                    branching,
                    data_format,
                    cache,
                    error_function,
                    heuristic,
                );
                GenericDL85::PurityDL85(learner)
            }
            SearchStrategy::NormalDL85 => {
                let learner = DL85::new(
                    min_sup,
                    max_depth,
                    max_error,
                    max_time,
                    one_time_sort,
                    cache_init_size,
                    cache_init_strategy,
                    specialization,
                    lower_bound_strategy,
                    branching,
                    data_format,
                    cache,
                    error_function,
                    heuristic,
                );
                GenericDL85::BaseDL85(learner)
            }
            _ => {
                panic!("This strategy is not allowed here.")
            }
        }
    }

    pub fn fit<S: Structure>(&mut self, structure: &mut S) {
        match self {
            GenericDL85::BaseDL85(ref mut learner) => learner.fit(structure),
            GenericDL85::RestartDL85(ref mut learner) => learner.fit(structure),
            GenericDL85::PurityDL85(ref mut learner) => learner.fit(structure),
        }
    }

    pub fn partial_fit<S: Structure>(&mut self, structure: &mut S, runtime: usize) {
        match self {
            GenericDL85::BaseDL85(ref mut learner) => learner.fit(structure),
            GenericDL85::RestartDL85(ref mut learner) => {
                let _ = learner.partial_fit(structure, Some(runtime));
            }
            GenericDL85::PurityDL85(ref mut learner) => {
                let _ = learner.partial_fit(structure, Some(runtime));
            }
        }
    }

    pub fn get_solution_tree(&self) -> Tree {
        match self {
            GenericDL85::BaseDL85(ref learner) => learner.tree.clone(),
            GenericDL85::RestartDL85(ref learner) => learner.tree.clone(),
            GenericDL85::PurityDL85(ref learner) => learner.tree.clone(),
        }
    }

    pub fn get_statistics(&self) -> Statistics {
        match self {
            GenericDL85::BaseDL85(ref learner) => learner.statistics,
            GenericDL85::RestartDL85(ref learner) => learner.statistics,
            GenericDL85::PurityDL85(ref learner) => learner.statistics,
        }
    }

    pub fn is_optimal(&self) -> bool {
        match self {
            GenericDL85::BaseDL85(ref learner) => learner.is_optimal(),
            GenericDL85::RestartDL85(ref learner) => learner.is_optimal(),
            GenericDL85::PurityDL85(ref learner) => learner.is_optimal(),
        }
    }
}
