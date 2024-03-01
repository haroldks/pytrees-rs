use crate::searches::{
    BranchingStrategy, CacheInitStrategy, CacheType, D2Objective, LowerBoundStrategy,
    SearchHeuristic, SearchStrategy, Specialization,
};
use clap::{arg, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[clap(name = "dt-trees", version, author, about)]
pub struct App {
    /// Dataset input file path
    #[clap(short, long, value_parser)]
    pub(crate) input: PathBuf,

    #[clap(subcommand)]
    pub(crate) command: ArgCommand,

    /// Printing Statistics and Constraints
    #[arg(long, default_value_t = false)]
    pub(crate) print_stats: bool,

    /// Printing Tree
    #[arg(long, default_value_t = false)]
    pub(crate) print_tree: bool,
}

#[derive(Debug, Subcommand)]
pub(crate) enum ArgCommand {
    /// DL8.5 Optimal search Algorithm with no depth limit and classification error as criterion.
    /// TODO : More arguments will be added to support LDS.
    dl85 {
        /// Minimum support
        #[arg(short, long, default_value_t = 1)]
        support: usize,

        /// Maximum depth
        #[arg(short, long)]
        depth: usize,

        /// Sorting Features based on heuristic only at the root (true) or at each node
        #[arg(long, default_value_t = true)]
        sorting_once: bool,

        /// Use Murtree Specialization Algorithm
        #[arg(long, value_enum, default_value_t = Specialization::None_)]
        specialization: Specialization,

        /// Lower bound heuristic strategy
        #[arg(long="lb", value_enum, default_value_t = LowerBoundStrategy::None_)]
        lower_bound_heuristic: LowerBoundStrategy,

        /// Branching type
        #[arg(short, long, value_enum, default_value_t = BranchingStrategy::None_)]
        branching: BranchingStrategy,

        #[arg(long="cache", value_enum, default_value_t = CacheType::Trie)]
        cache_type: CacheType,

        /// Cache init size
        /// Represents the reserved starting size of the cache
        #[arg(long, default_value_t = 0)]
        cache_init_size: usize,

        /// Cache Initialization strategy
        #[arg(long, value_enum, default_value_t = CacheInitStrategy::None_)]
        init_strategy: CacheInitStrategy,

        /// Sorting heuristic
        #[arg(short, long, value_enum, default_value_t = SearchHeuristic::None_)]
        heuristic: SearchHeuristic,

        /// Tree error initial upper bound
        #[arg(long, default_value_t = <f64>::INFINITY)]
        max_error: f64,

        /// Maximum time allowed to the search
        #[clap(long, short)]
        timeout: Option<usize>,
    },

    /// Optimal depth 2 algorithms using Error or Information as criterion
    d2_odt {
        /// Minimum support
        #[arg(short, long, default_value_t = 1)]
        support: usize,

        /// Depth
        /// The depth you want. The algorithm is optimized for depth 1 and 2 and won't work for more than that
        #[arg(short, long, default_value_t = 2)]
        depth: usize,

        /// Objective to optimise. Error or Information Gain
        #[arg(short, long, value_enum, default_value_t = D2Objective::Error)]
        objective: D2Objective,
    },

    /// Less greedy decision tree approach usind misclassification or information gain tree as sliding window
    lgdt {
        /// Minimum support
        #[arg(short, long, default_value_t = 1)]
        support: usize,

        /// Maximum depth
        #[arg(short, long)]
        depth: usize,

        /// Objective function inside
        #[arg(short, long, value_enum, default_value_t = D2Objective::Error)]
        objective: D2Objective,
    },
}
