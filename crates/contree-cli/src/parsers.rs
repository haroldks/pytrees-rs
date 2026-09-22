use clap::Parser;
use contree::algorithms::GenericConTree;
use contree::common::{PointSelector, ScheduleKind};
use std::path::PathBuf;

/// Command line arguments of `con-tree`.
#[derive(Debug, Parser)]
#[clap(name = "con-tree", version, author, about)]
pub struct GeneralParser {
    /// Dataset file: one instance per line, whitespace separated, label first
    #[clap(short, long, value_parser)]
    pub input: PathBuf,

    /// Minimum number of instances in each leaf
    #[arg(short, long, default_value_t = 1)]
    pub support: usize,

    /// Maximum depth of the tree
    #[arg(short, long)]
    pub depth: usize,

    /// Initial upper bound on the training error
    #[arg(long, default_value_t = usize::MAX)]
    pub max_error: usize,

    /// Error gap to the optimum that is tolerated (0 for an exact search)
    #[arg(long, default_value_t = 0)]
    pub max_gap: usize,

    /// Time limit in seconds
    #[arg(short, long, default_value_t = 600.0)]
    pub time_limit: f64,

    /// Explore features and thresholds in order of Gini impurity
    #[arg(long, default_value_t = false)]
    pub sort_by_heuristic: bool,

    /// Which threshold to evaluate next inside an interval: mid, first or random
    #[arg(
        long,
        default_value_t = PointSelector::Mid,
        value_parser = clap::value_parser!(PointSelector),
    )]
    pub split_selection_strategy: PointSelector,

    /// Use the specialised solver for depth-2 subtrees (the default)
    #[arg(short, long, default_value_t = false, hide = true)]
    pub fast_d2: bool,

    /// Disable the depth-2 solver and run the general search all the way down.
    /// Exact either way, but much slower
    #[arg(long, default_value_t = false, conflicts_with = "fast_d2")]
    pub no_fast_d2: bool,

    /// Use the anytime search (limited discrepancy search)
    #[arg(long, default_value_t = false)]
    pub use_lds: bool,

    /// How the anytime search (--use-lds) widens its budget between passes
    #[arg(
        long,
        default_value_t = ScheduleKind::Diagonal,
        value_parser = clap::value_parser!(ScheduleKind),
    )]
    pub budget_schedule: ScheduleKind,

    /// Print the search statistics and why the search stopped
    #[arg(long, default_value_t = false)]
    pub print_stats: bool,

    /// Print the tree
    #[arg(long, default_value_t = false)]
    pub print_tree: bool,

    /// Directory where the `lds` example writes its JSON results
    #[arg(long)]
    pub result_dir: Option<PathBuf>,

    /// Overwrite existing results in `result_dir`
    #[arg(long, default_value_t = false)]
    pub overwrite: bool,
}

impl From<GeneralParser> for GenericConTree {
    fn from(parser: GeneralParser) -> Self {
        GenericConTree::new(
            parser.support,
            parser.depth,
            parser.time_limit,
            parser.max_error,
            parser.split_selection_strategy,
            parser.max_gap,
            parser.sort_by_heuristic,
            !parser.no_fast_d2,
            parser.use_lds,
        )
        .with_budget_schedule(parser.budget_schedule)
    }
}

impl From<&GeneralParser> for GenericConTree {
    fn from(parser: &GeneralParser) -> Self {
        GenericConTree::new(
            parser.support,
            parser.depth,
            parser.time_limit,
            parser.max_error,
            parser.split_selection_strategy,
            parser.max_gap,
            parser.sort_by_heuristic,
            !parser.no_fast_d2,
            parser.use_lds,
        )
        .with_budget_schedule(parser.budget_schedule)
    }
}
