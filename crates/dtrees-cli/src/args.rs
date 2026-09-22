use clap::{arg, Parser, Subcommand};
use dtrees_rs::algorithms::common::types::{
    BranchingPolicy, LowerBoundPolicy, OptimalDepth2Policy, SearchHeuristic, SearchStrategy,
};
use std::path::PathBuf;

/// Command line arguments of `dtrees`.
#[derive(Debug, Parser)]
#[clap(name = "dt-trees", version, author, about)]
pub struct MainApp {
    /// Dataset file: one instance per line, label first, binary features
    #[clap(short, long, value_parser)]
    pub input: PathBuf,

    #[clap(subcommand)]
    pub command: ArgCommand,

    /// Print the search statistics
    #[arg(long, default_value_t = false)]
    pub print_stats: bool,

    /// Print the tree
    #[arg(long, default_value_t = false)]
    pub print_tree: bool,
}

/// The algorithm to run.
#[derive(Debug, Subcommand)]
pub enum ArgCommand {
    /// DL8.5: the optimal tree of a given depth, minimising the classification error
    DL85 {
        /// Minimum number of instances in each leaf
        #[arg(short, long, default_value_t = 1)]
        support: usize,

        /// Maximum depth of the tree
        #[arg(short, long)]
        depth: usize,

        /// Sort the features by the heuristic at every node (true) or only at the root
        #[arg(long, default_value_t = true)]
        always_sort: bool,

        /// Use the depth-2 solver for the last two levels
        #[arg(long, value_enum, default_value_t = OptimalDepth2Policy::Enabled)]
        depth2_policy: OptimalDepth2Policy,

        /// Lower bound used to prune subproblems
        #[arg(long="lb", value_enum, default_value_t = LowerBoundPolicy::Disabled)]
        lower_bound_policy: LowerBoundPolicy,

        /// Which branch of a feature is searched first
        #[arg(short, long, value_enum, default_value_t = BranchingPolicy::Default)]
        branching_policy: BranchingPolicy,

        /// Heuristic used to order the features
        #[arg(long, value_enum, default_value_t = SearchHeuristic::NoHeuristic)]
        heuristic: SearchHeuristic,

        /// Initial upper bound on the tree error
        #[arg(long, default_value_t = <f64>::INFINITY)]
        max_error: f64,

        /// Time limit in seconds
        #[clap(long, short)]
        timeout: Option<f64>,

        /// Print the configuration
        #[arg(long, default_value_t = false)]
        print_config: bool,
    },

    /// A tree of depth 1 or 2, minimising the error or maximising information gain
    D2 {
        /// Minimum number of instances in each leaf
        #[arg(short, long, default_value_t = 1)]
        support: usize,

        /// Depth of the tree: 1 or 2
        #[arg(short, long, default_value_t = 2)]
        depth: usize,

        /// Objective: error or information gain
        #[arg(short, long, value_enum, default_value_t = SearchStrategy::Depth2ErrorMinimizer)]
        objective: SearchStrategy,
    },

    /// LGDT: a greedy tree whose tests are chosen with a depth-2 lookahead
    Lgdt {
        /// Minimum number of instances in each leaf
        #[arg(short, long, default_value_t = 1)]
        support: usize,

        /// Maximum depth of the tree
        #[arg(short, long)]
        depth: usize,

        /// Objective of the depth-2 lookahead: error or information gain
        #[arg(short, long, value_enum, default_value_t = SearchStrategy::Depth2ErrorMinimizer)]
        objective: SearchStrategy,

        /// Print the configuration
        #[arg(long, default_value_t = false)]
        print_config: bool,
    },
}
