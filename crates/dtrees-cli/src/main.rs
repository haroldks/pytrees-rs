use clap::Parser;
use dtrees_rs::algorithms::common::errors::NativeError;
use dtrees_rs::algorithms::common::heuristics::Heuristic;
use dtrees_rs::algorithms::common::types::{
    CacheType, NodeDataType, SearchStatistics, SearchStrategy,
};
use dtrees_rs::algorithms::greedy::{LGDTBuilder, LGDT};
use dtrees_rs::algorithms::optimal::depth2::{
    ErrorMinimizer, InfoGainMaximizer, OptimalDepth2Tree,
};
use dtrees_rs::algorithms::optimal::dl85::DL85Builder;
use dtrees_rs::algorithms::TreeSearchAlgorithm;
use dtrees_rs::caching::{Caching, Trie};
use dtrees_rs::reader::data_reader::DataReader;
use dtrees_rs::tree::Tree;

mod args;

use args::{ArgCommand, MainApp};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = MainApp::parse();
    if !app.input.exists() {
        panic!("File does not exist");
    }

    let reader = DataReader::default();
    let mut cover = reader.read_file(&app.input)?;

    let mut statistics = SearchStatistics::default();
    let mut tree = Tree::default();

    match app.command {
        ArgCommand::D2 {
            support,
            depth,
            objective,
        } => {
            if depth == 0 || depth > 2 {
                panic!("Invalid depth depth must be 1 or 2");
            }

            let learner: Box<dyn OptimalDepth2Tree> = match objective {
                SearchStrategy::Depth2ErrorMinimizer => {
                    Box::<ErrorMinimizer<NativeError>>::default()
                }
                SearchStrategy::Depth2InfoGainMaximizer => {
                    Box::<InfoGainMaximizer<NativeError>>::default()
                }
                _ => {
                    panic!("Error wrong algorithm")
                }
            };

            tree = learner.fit(support, depth, &mut cover, None)?;
        }

        ArgCommand::Lgdt {
            support,
            depth,
            objective,
            print_config,
        } => {
            let obejective_fn: Box<dyn OptimalDepth2Tree> = match objective {
                SearchStrategy::Depth2ErrorMinimizer => {
                    Box::<ErrorMinimizer<NativeError>>::default()
                }
                SearchStrategy::Depth2InfoGainMaximizer => {
                    Box::<InfoGainMaximizer<NativeError>>::default()
                }
                _ => {
                    panic!("Error wrong objective method")
                }
            };

            let mut learner = LGDTBuilder::default()
                .min_support(support)
                .max_depth(depth)
                .search(obejective_fn)
                .build()?;

            learner.fit(&mut cover)?;
            tree = learner.tree().clone();

            if print_config {
                println!("{:#?}", learner.config())
            }
        }

        ArgCommand::DL85 {
            support,
            depth,
            always_sort,
            depth2_policy,
            lower_bound_policy,
            branching_policy,
            cache_type,
            heuristic,
            max_error,
            timeout,
            print_config,
        } => {
            let timeout = timeout.unwrap_or(f64::INFINITY);

            let heuristic_fn: Box<dyn Heuristic> = heuristic.into();
            let cache: Box<dyn Caching> = match cache_type {
                CacheType::Trie => Box::<Trie>::default(),
                CacheType::Hashmap => {
                    panic!("Not yet implemented")
                }
            };

            let depth2_search = Box::<ErrorMinimizer<NativeError>>::default();
            let error_fn = Box::<NativeError>::default();

            let mut learner = DL85Builder::default()
                .min_support(support)
                .max_depth(depth)
                .max_time(timeout)
                .max_error(max_error)
                .always_sort(always_sort)
                .cache(cache)
                .specialization(depth2_policy)
                .depth2_search(depth2_search)
                .error_function(error_fn)
                .heuristic(heuristic_fn)
                .lower_bound_strategy(lower_bound_policy)
                .branching_strategy(branching_policy)
                .node_exposed_data(NodeDataType::ClassesSupport)
                .build()?;

            learner.fit(&mut cover)?;

            statistics = *learner.statistics();
            tree = learner.tree().clone();

            if print_config {
                println!("{:#?}", learner.config())
            }
        }
    }

    if app.print_stats {
        println!("{:#?}", statistics);
    }

    if app.print_tree {
        tree.print();
    }

    Ok(())
}
