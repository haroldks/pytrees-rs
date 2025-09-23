pub mod examples;

use crate::algorithms::common::types::{
    BranchingPolicy, CacheType, LowerBoundPolicy, OptimalDepth2Policy, SearchHeuristic,
    SearchStrategy,
};
use clap::{arg, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[clap(name = "dt-trees", version, author, about)]
pub struct MainApp {
    /// Dataset input file path
    #[clap(short, long, value_parser)]
    pub input: PathBuf,

    #[clap(subcommand)]
    pub command: ArgCommand,

    /// Printing Statistics and Constraints
    #[arg(long, default_value_t = false)]
    pub print_stats: bool,

    /// Printing Tree
    #[arg(long, default_value_t = false)]
    pub print_tree: bool,
}

#[derive(Debug, Subcommand)]
pub enum ArgCommand {
    /// DL8.5 Optimal search Algorithm with no depth limit and classification error as criterion.
    /// TODO : More arguments will be added to support LDS.
    DL85 {
        /// Minimum support
        #[arg(short, long, default_value_t = 1)]
        support: usize,

        /// Maximum depth
        #[arg(short, long)]
        depth: usize,

        /// Sorting Features based on heuristic only at the root (true) or at each node
        #[arg(long, default_value_t = true)]
        always_sort: bool,

        /// Use Murtree Specialization Algorithm
        #[arg(long, value_enum, default_value_t = OptimalDepth2Policy::Enabled)]
        depth2_policy: OptimalDepth2Policy,

        /// Lower bound heuristic strategy
        #[arg(long="lb", value_enum, default_value_t = LowerBoundPolicy::Disabled)]
        lower_bound_policy: LowerBoundPolicy,

        /// Branching type
        #[arg(short, long, value_enum, default_value_t = BranchingPolicy::Default)]
        branching_policy: BranchingPolicy,

        #[arg(long="cache", value_enum, default_value_t = CacheType::Trie)]
        cache_type: CacheType,

        /// Sorting heuristic
        #[arg(long, value_enum, default_value_t = SearchHeuristic::NoHeuristic)]
        heuristic: SearchHeuristic,

        /// Tree error initial upper bound
        #[arg(long, default_value_t = <f64>::INFINITY)]
        max_error: f64,

        /// Maximum time allowed to the search
        #[clap(long, short)]
        timeout: Option<f64>,

        /// Printing Config
        #[arg(long, default_value_t = false)]
        print_config: bool,
    },

    /// Optimal depth 2 algorithms using Error or Information as criterion
    D2 {
        /// Minimum support
        #[arg(short, long, default_value_t = 1)]
        support: usize,

        /// Depth
        /// The depth you want. The algorithm is optimized for depth 1 and 2 and won't work for more than that
        #[arg(short, long, default_value_t = 2)]
        depth: usize,

        /// Objective to optimise. Error or Information Gain
        #[arg(short, long, value_enum, default_value_t = SearchStrategy::Depth2ErrorMinimizer)]
        objective: SearchStrategy,
    },

    /// Less greedy decision tree approach using misclassification or information gain tree as sliding window
    Lgdt {
        /// Minimum support
        #[arg(short, long, default_value_t = 1)]
        support: usize,

        /// Maximum depth
        #[arg(short, long)]
        depth: usize,

        /// Objective function inside
        #[arg(short, long, value_enum, default_value_t = SearchStrategy::Depth2ErrorMinimizer)]
        objective: SearchStrategy,

        /// Printing Config
        #[arg(long, default_value_t = false)]
        print_config: bool,
    },
}
