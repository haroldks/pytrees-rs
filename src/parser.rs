use crate::algorithms::common::types::{OptimalDepth2Policy, SearchHeuristic, SearchStepStrategy};
use crate::tree::Tree;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::{remove_file, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;

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

    /// The time limit for the run
    #[arg(short, long, default_value_t = 300.0)]
    pub timeout: f64,

    /// Metric
    #[arg(short, long, default_value_t = 1.0)]
    pub metric: f64,

    /// Metric
    #[arg(long, default_value_t = 0.002)]
    pub epsilon: f64,

    /// Printing Statistics and Constraints
    #[arg(long, value_enum, default_value_t = OptimalDepth2Policy::Enabled)]
    pub fast_d2: OptimalDepth2Policy,

    /// Printing Statistics and Constraints
    #[arg(long, default_value_t = false)]
    pub print_stats: bool,

    /// Printing Statistics and Constraints
    #[arg(long, default_value_t = true)]
    pub always_sort: bool,

    /// Sorting heuristic
    #[arg(long, value_enum, default_value_t = SearchHeuristic::None_)]
    pub heuristic: SearchHeuristic,

    /// Discrepancy Strategy
    #[arg(long, value_enum, default_value_t = SearchStepStrategy::Monotonic)]
    pub step: SearchStepStrategy,

    /// Result_dir
    #[arg(short, long)]
    pub result: PathBuf,

    /// Printing Tree
    #[arg(long, default_value_t = false)]
    pub print_tree: bool,

    /// Overwriting file
    #[arg(long, default_value_t = false)]
    pub overwrite: bool,
}

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

pub fn save_results(result: &Res, result_path: &PathBuf) -> std::io::Result<()> {
    // Create parent directories if they don't exist
    if let Some(parent) = result_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let file = File::create(result_path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, result)?;
    writer.flush()
}

pub fn load_results(result_path: &PathBuf) -> Option<Res> {
    if !result_path.exists() {
        return None;
    }

    File::open(result_path).ok().and_then(|file| {
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).ok()
    })
}

pub fn remove_results(result_path: &PathBuf) -> std::io::Result<()> {
    if result_path.exists() {
        remove_file(result_path)?
    }
    Ok(())
}
