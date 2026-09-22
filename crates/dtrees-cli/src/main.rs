use clap::Parser;
use dtrees_rs::algorithms::common::errors::NativeError;
use dtrees_rs::algorithms::common::heuristics::Heuristic;
use dtrees_rs::algorithms::common::types::{NodeDataType, SearchStatistics};
use dtrees_rs::algorithms::greedy::LGDTBuilder;
use dtrees_rs::algorithms::optimal::depth2::{
    ErrorMinimizer, InfoGainMaximizer, OptimalDepth2Tree,
};
use dtrees_rs::algorithms::optimal::dl85::DL85Builder;
use dtrees_rs::algorithms::TreeSearchAlgorithm;
use dtrees_rs::caching::{Caching, Trie};
use dtrees_rs::reader::data_reader::DataReader;
use dtrees_rs::tree::Tree;
use std::process::ExitCode;

mod args;

use args::{ArgCommand, MainApp, Objective};

fn main() -> ExitCode {
    match run(MainApp::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run(app: MainApp) -> Result<(), Box<dyn std::error::Error>> {
    if !app.input.exists() {
        return Err(format!("{} does not exist", app.input.display()).into());
    }

    let reader = DataReader::default();
    let mut cover = reader.read_file(&app.input)?;

    let mut statistics = SearchStatistics::default();
    let tree: Tree;

    match app.command {
        ArgCommand::D2 {
            support,
            depth,
            objective,
        } => {
            if depth == 0 || depth > 2 {
                return Err(format!("d2 needs a depth of 1 or 2, not {depth}").into());
            }

            tree = depth2_solver(objective).fit(support, depth, &mut cover, None)?;
        }

        ArgCommand::Lgdt {
            support,
            depth,
            objective,
            print_config,
        } => {
            let mut learner = LGDTBuilder::default()
                .min_support(support)
                .max_depth(depth)
                .search(depth2_solver(objective))
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
            heuristic,
            max_error,
            timeout,
            print_config,
        } => {
            let timeout = timeout.unwrap_or(f64::INFINITY);

            let heuristic_fn: Box<dyn Heuristic> = heuristic.into();
            let cache: Box<dyn Caching> = Box::<Trie>::default();

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
        println!("{}", tree);
    }

    Ok(())
}

/// The depth-2 solver for `objective`.
fn depth2_solver(objective: Objective) -> Box<dyn OptimalDepth2Tree> {
    match objective {
        Objective::Error => Box::<ErrorMinimizer<NativeError>>::default(),
        Objective::InformationGain => Box::<InfoGainMaximizer<NativeError>>::default(),
    }
}
