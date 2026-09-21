use clap::Parser;
use contree::algorithms::GenericConTree;
use contree::common::{PointSelector, ScheduleKind};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[clap(name = "con-tree", version, author, about)]
pub struct GeneralParser {
    /// Dataset input file path
    #[clap(short, long, value_parser)]
    pub input: PathBuf,

    /// Minimum support
    #[arg(short, long, default_value_t = 1)]
    pub support: usize,

    /// Maximum depth
    #[arg(short, long)]
    pub depth: usize,

    /// Maximum error allowed
    #[arg(long, default_value_t = usize::MAX)]
    pub max_error: usize,

    /// Maximum error gap allowed
    #[arg(long, default_value_t = 0)]
    pub max_gap: usize,

    /// Maximum execution time allowed
    #[arg(short, long, default_value_t = 600.0)]
    pub time_limit: f64,

    /// Sort split and feature using gini index
    #[arg(long, default_value_t = false)]
    pub sort_by_heuristic: bool,

    /// Split selection strategy to use in the search
    #[arg(
        long,
        default_value_t = PointSelector::Mid,
        value_parser = clap::value_parser!(PointSelector),
    )]
    pub split_selection_strategy: PointSelector,

    /// Use the specialized solver for depth-2 subtrees. This is the default;
    /// the flag is kept so existing scripts that pass it keep working.
    #[arg(short, long, default_value_t = false, hide = true)]
    pub fast_d2: bool,

    /// Disable the specialized depth-2 solver and run the general search all
    /// the way down. Exact either way, but much slower at depth 2 and beyond.
    #[arg(long, default_value_t = false, conflicts_with = "fast_d2")]
    pub no_fast_d2: bool,

    /// Use LDS
    #[arg(long, default_value_t = false)]
    pub use_lds: bool,

    /// How the anytime search (--use-lds) widens its budget between passes
    #[arg(
        long,
        default_value_t = ScheduleKind::Diagonal,
        value_parser = clap::value_parser!(ScheduleKind),
    )]
    pub budget_schedule: ScheduleKind,

    /// Printing Statistics and Constraints
    #[arg(long, default_value_t = false)]
    pub print_stats: bool,

    /// Printing Tree
    #[arg(long, default_value_t = false)]
    pub print_tree: bool,

    #[arg(long)]
    pub result_dir: PathBuf,

    /// Overwriting file
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
