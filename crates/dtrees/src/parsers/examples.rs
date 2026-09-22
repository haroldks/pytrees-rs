//! Command line arguments and result files shared by the anytime examples.

use crate::algorithms::common::types::{OptimalDepth2Policy, SearchHeuristic, SearchStepStrategy};
use crate::tree::Tree;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::{remove_file, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;

/// Command line arguments of the anytime examples.
#[derive(Debug, Parser)]
#[clap(name = "dt-trees", version, author, about)]
pub struct ExampleParser {
    /// Dataset file: one instance per line, label first, binary features
    #[clap(short, long, value_parser)]
    pub input: PathBuf,

    /// Minimum number of instances in each leaf
    #[arg(short, long, default_value_t = 5)]
    pub support: usize,

    /// Maximum depth of the tree
    #[arg(short, long)]
    pub depth: usize,

    /// Time limit in seconds
    #[arg(short, long, default_value_t = 300.0)]
    pub timeout: f64,

    /// Not used by the current examples
    #[arg(short, long, default_value_t = 1.0)]
    pub metric: f64,

    /// Step of the gain and purity rules
    #[arg(long, default_value_t = 0.002)]
    pub epsilon: f64,

    /// Whether to use the depth-2 solver
    #[arg(long, value_enum, default_value_t = OptimalDepth2Policy::Enabled)]
    pub fast_d2: OptimalDepth2Policy,

    /// Print the search statistics
    #[arg(long, default_value_t = false)]
    pub print_stats: bool,

    /// Sort the features by the heuristic at every node
    #[arg(long, default_value_t = true)]
    pub always_sort: bool,

    /// Heuristic used to order the features
    #[arg(long, value_enum, default_value_t = SearchHeuristic::NoHeuristic)]
    pub heuristic: SearchHeuristic,

    /// How the budget of the search rule grows between passes
    #[arg(long, value_enum, default_value_t = SearchStepStrategy::Monotonic)]
    pub step: SearchStepStrategy,

    /// Directory where the JSON results are written
    #[arg(short, long)]
    pub result: PathBuf,

    /// Print the tree
    #[arg(long, default_value_t = false)]
    pub print_tree: bool,

    /// Overwrite existing results
    #[arg(long, default_value_t = false)]
    pub overwrite: bool,
}

/// The results of one run, saved as JSON.
#[derive(Serialize, Deserialize, Clone)]
pub struct Res {
    pub name: String,
    pub method: String,
    pub depth: usize,
    pub support: usize,
    pub completed: bool,
    pub one_time_sort: bool,
    pub fast_d2: bool,
    pub metric: Vec<f64>,
    pub runtimes: Vec<f64>,
    pub errors: Vec<f64>,
    pub cache: Vec<usize>,
    pub tree: Tree,
}

/// Writes `result` to `result_path`, creating its directory.
pub fn save_results(result: &Res, result_path: &PathBuf) -> std::io::Result<()> {
    if let Some(parent) = result_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let file = File::create(result_path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, result)?;
    writer.flush()
}

/// Reads results saved earlier, if the file exists and parses.
pub fn load_results(result_path: &PathBuf) -> Option<Res> {
    if !result_path.exists() {
        return None;
    }

    File::open(result_path).ok().and_then(|file| {
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).ok()
    })
}

/// Deletes a result file if it exists.
pub fn remove_results(result_path: &PathBuf) -> std::io::Result<()> {
    if result_path.exists() {
        remove_file(result_path)?
    }
    Ok(())
}
