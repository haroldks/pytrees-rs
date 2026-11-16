use clap::Parser;
use contree::algorithms::GenericConTree;
use contree::common::{Deepening, PointSelector, ScheduleKind};
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
    #[arg(long, default_value_t = f64::INFINITY)]
    pub max_error: f64,

    /// Error gap to the optimum that is tolerated (0 for an exact search)
    #[arg(long, default_value_t = 0.0)]
    pub max_gap: f64,

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

    /// How the anytime search (--use-lds) widens its budget between passes.
    /// Defaults to diagonal with the `first` selector and square with the
    /// others: only `first` uses the split budget that diagonal also widens
    #[arg(long, value_parser = clap::value_parser!(ScheduleKind))]
    pub budget_schedule: Option<ScheduleKind>,

    /// SPLIT: search only the top LOOKAHEAD levels and complete each node
    /// below them with a greedy tree. Runs the anytime search, with the
    /// discrepancy and split budgets only if --use-lds is given
    #[arg(long)]
    pub lookahead: Option<usize>,

    /// With --lookahead: go one level deeper between passes until the
    /// lookahead covers the depth, which ends on the optimal tree
    #[arg(long, default_value_t = false, requires = "lookahead")]
    pub anytime: bool,

    /// With --anytime and --use-lds: spread the deepenings so that the
    /// lookahead covers the depth when the budget schedule reaches its last
    /// budget, instead of going deeper after every pass
    #[arg(long, default_value_t = false, requires = "anytime")]
    pub sync: bool,

    /// With --lookahead: run the interval pruner only in passes whose
    /// lookahead covers the depth. Its rules assume optimal subtree costs, so
    /// in other passes it may skip a threshold whose greedy completion is
    /// better
    #[arg(long, default_value_t = false, requires = "lookahead")]
    pub no_lookahead_pruning: bool,

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

impl GeneralParser {
    /// A warning when the budget schedule chosen does not suit the split
    /// selector; see [`ScheduleKind::mismatch`].
    pub fn schedule_warning(&self) -> Option<String> {
        if !self.use_lds {
            return None;
        }
        self.budget_schedule?
            .mismatch(self.split_selection_strategy)
    }

    /// How the lookahead deepens between passes.
    pub fn deepening(&self) -> Deepening {
        match (self.anytime, self.use_lds && self.sync) {
            (false, _) => Deepening::Fixed,
            (true, true) => Deepening::Synced,
            (true, false) => Deepening::EveryPass,
        }
    }
}

impl From<GeneralParser> for GenericConTree {
    fn from(parser: GeneralParser) -> Self {
        GenericConTree::from(&parser)
    }
}

impl From<&GeneralParser> for GenericConTree {
    fn from(parser: &GeneralParser) -> Self {
        let solver = GenericConTree::new(
            parser.support,
            parser.depth,
            parser.time_limit,
            parser.max_error,
            parser.split_selection_strategy,
            parser.max_gap,
            parser.sort_by_heuristic,
            !parser.no_fast_d2,
            parser.use_lds || parser.lookahead.is_some(),
        );
        let solver = match parser.budget_schedule {
            Some(schedule) => solver.with_budget_schedule(schedule),
            None => solver,
        };
        match parser.lookahead {
            Some(lookahead) => solver
                .with_lds(parser.use_lds)
                .with_lookahead(lookahead, parser.deepening())
                .with_lookahead_pruning(!parser.no_lookahead_pruning),
            None => solver,
        }
    }
}
