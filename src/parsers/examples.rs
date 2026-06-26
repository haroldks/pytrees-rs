#![allow(dead_code)]
use crate::algorithms::common::types::{
    CacheType, OptimalDepth2Policy, SearchHeuristic, SearchStatistics, SearchStepStrategy,
};
use crate::algorithms::optimal::rules::{
    DiscrepancyRule, Exponential, GainRule, Luby, Monotonic, TopkRule,
};
use crate::algorithms::optimal::IterativeSearch;
use crate::algorithms::optimal::Reason;
use crate::cover::Cover;
use crate::tree::Tree;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::{remove_file, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[clap(name = "dt-trees", version, author, about)]
pub struct ExampleParser {
    /// Dataset input file path
    #[clap(short, long, value_parser)]
    pub input: PathBuf,

    /// Minimum support
    #[arg(short, long, default_value_t = 5)]
    pub support: usize,

    /// Maximum depth
    #[arg(short, long)]
    pub depth: usize,

    /// Time limit in seconds
    #[arg(short, long, default_value_t = 300.0)]
    pub timeout: f64,

    /// Convergence epsilon for gain/purity rules
    #[arg(long, default_value_t = 0.002)]
    pub epsilon: f64,

    /// Enable depth-2 specialization
    #[arg(long, value_enum, default_value_t = OptimalDepth2Policy::Enabled)]
    pub fast_d2: OptimalDepth2Policy,

    /// Print search statistics after completion
    #[arg(long, default_value_t = false)]
    pub print_stats: bool,

    /// Sort candidates on every node visit
    #[arg(long, default_value_t = true)]
    pub always_sort: bool,

    /// Sorting heuristic
    #[arg(long, value_enum, default_value_t = SearchHeuristic::NoHeuristic)]
    pub heuristic: SearchHeuristic,

    /// Step strategy for iterative search rules
    #[arg(long, value_enum, default_value_t = SearchStepStrategy::Monotonic)]
    pub step: SearchStepStrategy,

    /// Directory to store result files
    #[arg(short, long)]
    pub result: PathBuf,

    /// Print the resulting tree after completion
    #[arg(long, default_value_t = false)]
    pub print_tree: bool,

    /// Regularization parameter (lambda)
    #[arg(long, default_value_t = 0.0)]
    pub lambda: f64,

    /// Lookahead depth for split-based search
    #[arg(long, default_value_t = 0)]
    pub lookahead_depth: usize,

    /// Overwrite an existing completed result file
    #[arg(long, default_value_t = false)]
    pub overwrite: bool,

    /// Cache backend to use
    #[arg(long, value_enum, default_value_t = CacheType::Trie)]
    pub cache_type: CacheType,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Res {
    pub name: String,
    pub method: String,
    pub regularization: f64,
    pub lookahead_depth: Option<usize>,
    pub depth: usize,
    pub support: usize,
    pub completed: bool,
    pub one_time_sort: bool,
    pub fast_d2: bool,
    pub metric: Vec<f64>,
    pub runtimes: Vec<f64>,
    pub errors: Vec<f64>,
    pub lambdas: Vec<f64>,
    pub cache: Vec<usize>,
    pub tree: Tree,
}

impl ExampleParser {
    pub fn print_outcome(&self, stats: SearchStatistics, tree: &Tree) {
        if self.print_stats {
            println!("{:#?}", stats);
        }
        if self.print_tree {
            tree.print();
        }
    }

    pub fn print_tree_if(&self, tree: &Tree) {
        if self.print_tree {
            tree.print();
        }
    }

    pub fn fresh_result(&self, file: &str, method: &str) -> Res {
        Res {
            name: file.to_string(),
            method: method.to_string(),
            regularization: self.lambda,
            depth: self.depth,
            support: self.support,
            lookahead_depth: None,
            metric: Vec::with_capacity(100),
            runtimes: Vec::with_capacity(100),
            errors: Vec::with_capacity(100),
            lambdas: Vec::with_capacity(100),
            cache: Vec::with_capacity(100),
            completed: false,
            one_time_sort: !self.always_sort,
            tree: Default::default(),
            fast_d2: self.fast_d2 == OptimalDepth2Policy::Enabled,
        }
    }
}

pub fn make_gain_rule(strategy: SearchStepStrategy, epsilon: f64, depth: f64) -> GainRule {
    match strategy {
        SearchStepStrategy::Monotonic => {
            GainRule::new(0.0, epsilon, depth, Box::<Monotonic>::default())
        }
        SearchStepStrategy::Exponential => {
            GainRule::new(0.0, epsilon, depth, Box::<Exponential>::default())
        }
        SearchStepStrategy::Luby => GainRule::new(0.0, epsilon, depth, Box::<Luby>::default()),
    }
}

pub fn make_topk_rule(strategy: SearchStepStrategy, num_attributes: usize) -> TopkRule {
    match strategy {
        SearchStepStrategy::Monotonic => TopkRule::new(num_attributes, Box::<Monotonic>::default()),
        SearchStepStrategy::Exponential => {
            TopkRule::new(num_attributes, Box::<Exponential>::default())
        }
        SearchStepStrategy::Luby => TopkRule::new(num_attributes, Box::<Luby>::default()),
    }
}

pub fn make_discrepancy_rule(strategy: SearchStepStrategy) -> DiscrepancyRule {
    match strategy {
        SearchStepStrategy::Monotonic => {
            DiscrepancyRule::new(usize::MAX, Box::<Monotonic>::default())
        }
        SearchStepStrategy::Exponential => {
            DiscrepancyRule::new(usize::MAX, Box::<Exponential>::default())
        }
        SearchStepStrategy::Luby => DiscrepancyRule::new(usize::MAX, Box::<Luby>::default()),
    }
}

pub fn save_results(result: &Res, result_path: &Path) -> std::io::Result<()> {
    if let Some(parent) = result_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = File::create(result_path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, result)?;
    writer.flush()
}

pub fn load_results(result_path: &Path) -> Option<Res> {
    File::open(result_path)
        .ok()
        .and_then(|file| serde_json::from_reader(BufReader::new(file)).ok())
}

pub fn load_or_create_result(result_path: &Path, overwrite: bool, fresh: Res) -> Res {
    match load_results(result_path) {
        Some(res) if !res.completed => res,
        prev => {
            if prev.is_some() {
                if !overwrite {
                    eprintln!("Computation was already completed. Use different parameters or remove the result file to recompute.");
                } else if let Err(e) = remove_file(result_path) {
                    eprintln!("Warning: could not remove result file: {e}");
                }
            }
            fresh
        }
    }
}

pub fn run_iterative(
    algo: &mut dyn IterativeSearch,
    cover: &mut Cover,
    result: &mut Res,
    result_path: &Path,
    checkpoint_interval: usize,
) -> std::io::Result<SearchStatistics> {
    let mut counter = 0;
    while !algo.time_is_exhausted() {
        let r = algo.partial_fit(cover);
        let stats = algo.statistics();
        result.errors.push(stats.tree_error);
        result.cache.push(stats.cache_size);
        result.runtimes.push(stats.duration);
        result.lambdas.push(algo.tree().root_lambda());
        result.tree = algo.tree().clone();
        if counter > 0 && counter % checkpoint_interval == 0 {
            let _ = save_results(result, result_path);
        }
        counter += 1;
        if r.reason == Reason::Done {
            result.completed = true;
            break;
        }
    }
    result.tree = algo.tree().clone();
    save_results(result, result_path)?;
    Ok(*algo.statistics())
}
