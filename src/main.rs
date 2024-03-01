use crate::cache::trie::Trie;
use crate::cache::Caching;
use crate::data::{BinaryData, FileReader};
use crate::heuristics::{GiniIndex, Heuristic, InformationGain, InformationGainRatio, NoHeuristic};
use crate::parser::{App, ArgCommand};
use crate::searches::errors::NativeError;
use crate::searches::greedy::LGDT;
use crate::searches::optimal::d2::GenericDepth2;
use crate::searches::optimal::DL85;
use crate::searches::{
    CacheType, D2Objective, NodeExposedData, SearchHeuristic, SearchStrategy, Statistics,
};
use crate::structures::RevBitset;
use crate::tree::Tree;
use clap::Parser;

mod cache;
mod data;
mod globals;
mod heuristics;
mod parser;
mod searches;
mod structures;
mod tree;

fn main() {
    let app = App::parse();

    if !app.input.exists() {
        panic!("File does not exist");
    }

    let file = app.input.to_str().unwrap();
    let data = BinaryData::read(file, false, 0.0);
    let mut structure = RevBitset::new(&data);

    let mut statistics = Statistics::default();
    let mut tree = Tree::default();

    match app.command {
        ArgCommand::d2_odt {
            support,
            depth,
            objective,
        } => {
            if depth == 0 || depth > 2 {
                panic!("Invalid depth depth must be 1 or 2");
            }
            let strategy = match objective {
                D2Objective::Error => SearchStrategy::LessGreedyMurtree,
                D2Objective::InformationGain => SearchStrategy::LessGreedyInfoGain,
            };

            let mut learner = GenericDepth2::new(strategy);
            tree = learner.fit(support, depth, &mut structure);
        }

        ArgCommand::lgdt {
            support,
            depth,
            objective,
        } => {
            let strategy = match objective {
                D2Objective::Error => SearchStrategy::LessGreedyMurtree,
                D2Objective::InformationGain => SearchStrategy::LessGreedyInfoGain,
            };

            let mut learner = LGDT::new(support, depth, strategy);
            learner.fit(&mut structure);
            statistics = learner.statistics;
            tree = learner.tree.clone();
        }

        ArgCommand::dl85 {
            support,
            depth,
            sorting_once,
            specialization,
            lower_bound_heuristic,
            branching,
            cache_type,
            cache_init_size,
            init_strategy,
            heuristic,
            max_error,
            timeout,
        } => {
            let timeout = match timeout {
                None => <usize>::MAX,
                Some(t) => t,
            };

            let heuristic_fn: Box<dyn Heuristic> = match heuristic {
                SearchHeuristic::None_ => Box::<NoHeuristic>::default(),
                SearchHeuristic::InformationGain => Box::<InformationGain>::default(),
                SearchHeuristic::InformationGainRatio => Box::<InformationGainRatio>::default(),
                SearchHeuristic::GiniIndex => Box::<GiniIndex>::default(),
            };
            let cache: Box<dyn Caching> = match cache_type {
                CacheType::Trie => Box::<Trie>::default(),
                CacheType::Hashmap => {
                    panic!("Not yet implemented")
                }
            };

            let mut learner = DL85::new(
                support,
                depth,
                max_error,
                timeout,
                sorting_once,
                cache_init_size,
                init_strategy,
                specialization,
                lower_bound_heuristic,
                branching,
                NodeExposedData::ClassesSupport,
                cache,
                Box::<NativeError>::default(),
                heuristic_fn,
            );

            learner.fit(&mut structure);

            statistics = learner.statistics;
            tree = learner.tree.clone();
        }
    }

    if app.print_stats {
        println!("{:#?}", statistics);
    }

    if app.print_tree {
        tree.print();
    }
}
