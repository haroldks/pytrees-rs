//! Pins the error found and the effort spent (nodes searched, cache entries)
//! on a few datasets. On a mismatch, the test prints the new table.

use std::path::Path;

use dtrees_rs::algorithms::common::errors::NativeError;
use dtrees_rs::algorithms::common::heuristics::{Heuristic, InformationGain};
use dtrees_rs::algorithms::common::types::{
    BranchingPolicy, LowerBoundPolicy, OptimalDepth2Policy,
};
use dtrees_rs::algorithms::optimal::depth2::ErrorMinimizer;
use dtrees_rs::algorithms::optimal::dl85::DL85Builder;
use dtrees_rs::algorithms::optimal::rules::{DiscrepancyRule, Monotonic, Rule, TopkRule};
use dtrees_rs::algorithms::TreeSearchAlgorithm;
use dtrees_rs::caching::{Caching, CoverMap, Trie};
use dtrees_rs::reader::data_reader::DataReader;

const DATASETS: [&str; 5] = ["lymph", "primary-tumor", "soybean", "tic-tac-toe", "vote"];
/// `(config, max depth)`.
const CONFIGS: [(&str, usize); 6] = [
    ("plain", 2),
    ("lds", 2),
    ("d2", 3),
    ("d2+dyn+sim", 3),
    ("d2+lds", 3),
    ("d2+dyn+sim+topk", 3),
];
const CACHES: [&str; 2] = ["trie", "cover"];

/// `(dataset, config, depth, cache, error, nodes searched, cache entries)`, at
/// minimum support 1.
type Row = (
    &'static str,
    &'static str,
    usize,
    &'static str,
    f64,
    usize,
    usize,
);
const EXPECTED: &[Row] = &[
    ("lymph", "plain", 2, "trie", 22.0, 8460, 5275),
    ("lymph", "plain", 2, "cover", 22.0, 7759, 2491),
    ("lymph", "lds", 2, "trie", 22.0, 480554, 5275),
    ("lymph", "lds", 2, "cover", 22.0, 418601, 2491),
    ("lymph", "d2", 3, "trie", 12.0, 98, 734),
    ("lymph", "d2", 3, "cover", 12.0, 98, 433),
    ("lymph", "d2+dyn+sim", 3, "trie", 12.0, 98, 734),
    ("lymph", "d2+dyn+sim", 3, "cover", 12.0, 91, 414),
    ("lymph", "d2+lds", 3, "trie", 12.0, 3585, 734),
    ("lymph", "d2+lds", 3, "cover", 12.0, 3585, 433),
    ("lymph", "d2+dyn+sim+topk", 3, "trie", 12.0, 2445, 734),
    ("lymph", "d2+dyn+sim+topk", 3, "cover", 12.0, 2438, 414),
    ("primary-tumor", "plain", 2, "trie", 58.0, 2706, 1499),
    ("primary-tumor", "plain", 2, "cover", 58.0, 2221, 533),
    ("primary-tumor", "lds", 2, "trie", 58.0, 64797, 1499),
    ("primary-tumor", "lds", 2, "cover", 58.0, 53282, 533),
    ("primary-tumor", "d2", 3, "trie", 46.0, 57, 443),
    ("primary-tumor", "d2", 3, "cover", 46.0, 57, 207),
    ("primary-tumor", "d2+dyn+sim", 3, "trie", 46.0, 57, 443),
    ("primary-tumor", "d2+dyn+sim", 3, "cover", 46.0, 43, 173),
    ("primary-tumor", "d2+lds", 3, "trie", 46.0, 941, 443),
    ("primary-tumor", "d2+lds", 3, "cover", 46.0, 941, 207),
    (
        "primary-tumor",
        "d2+dyn+sim+topk",
        3,
        "trie",
        46.0,
        554,
        443,
    ),
    (
        "primary-tumor",
        "d2+dyn+sim+topk",
        3,
        "cover",
        46.0,
        540,
        173,
    ),
    ("soybean", "plain", 2, "trie", 55.0, 4069, 2490),
    ("soybean", "plain", 2, "cover", 55.0, 3960, 1713),
    ("soybean", "lds", 2, "trie", 55.0, 178616, 2426),
    ("soybean", "lds", 2, "cover", 55.0, 174366, 1665),
    ("soybean", "d2", 3, "trie", 29.0, 71, 594),
    ("soybean", "d2", 3, "cover", 29.0, 71, 307),
    ("soybean", "d2+dyn+sim", 3, "trie", 29.0, 71, 594),
    ("soybean", "d2+dyn+sim", 3, "cover", 29.0, 67, 305),
    ("soybean", "d2+lds", 3, "trie", 29.0, 1804, 594),
    ("soybean", "d2+lds", 3, "cover", 29.0, 1804, 307),
    ("soybean", "d2+dyn+sim+topk", 3, "trie", 29.0, 1347, 594),
    ("soybean", "d2+dyn+sim+topk", 3, "cover", 29.0, 1343, 305),
    ("tic-tac-toe", "plain", 2, "trie", 282.0, 2583, 1414),
    ("tic-tac-toe", "plain", 2, "cover", 282.0, 2583, 1360),
    ("tic-tac-toe", "lds", 2, "trie", 282.0, 51547, 1414),
    ("tic-tac-toe", "lds", 2, "cover", 282.0, 51547, 1360),
    ("tic-tac-toe", "d2", 3, "trie", 216.0, 50, 391),
    ("tic-tac-toe", "d2", 3, "cover", 216.0, 50, 315),
    ("tic-tac-toe", "d2+dyn+sim", 3, "trie", 216.0, 50, 391),
    ("tic-tac-toe", "d2+dyn+sim", 3, "cover", 216.0, 50, 315),
    ("tic-tac-toe", "d2+lds", 3, "trie", 216.0, 767, 391),
    ("tic-tac-toe", "d2+lds", 3, "cover", 216.0, 767, 315),
    ("tic-tac-toe", "d2+dyn+sim+topk", 3, "trie", 216.0, 429, 391),
    (
        "tic-tac-toe",
        "d2+dyn+sim+topk",
        3,
        "cover",
        216.0,
        429,
        315,
    ),
    ("vote", "plain", 2, "trie", 17.0, 4626, 3191),
    ("vote", "plain", 2, "cover", 17.0, 4626, 3122),
    ("vote", "lds", 2, "trie", 17.0, 165083, 3044),
    ("vote", "lds", 2, "cover", 17.0, 165083, 2981),
    ("vote", "d2", 3, "trie", 12.0, 71, 573),
    ("vote", "d2", 3, "cover", 12.0, 71, 430),
    ("vote", "d2+dyn+sim", 3, "trie", 12.0, 71, 573),
    ("vote", "d2+dyn+sim", 3, "cover", 12.0, 71, 430),
    ("vote", "d2+lds", 3, "trie", 12.0, 1918, 573),
    ("vote", "d2+lds", 3, "cover", 12.0, 1918, 430),
    ("vote", "d2+dyn+sim+topk", 3, "trie", 12.0, 1248, 573),
    ("vote", "d2+dyn+sim+topk", 3, "cover", 12.0, 1248, 430),
];

fn solve(dataset: &str, config: &str, depth: usize, cache: &str) -> (f64, usize, usize) {
    let path = format!("test_data/{dataset}.txt");
    let mut cover = DataReader::default()
        .read_file(Path::new(&path))
        .expect("the dataset reads");
    let error_fn = Box::<NativeError>::default();
    let heuristic: Box<dyn Heuristic> = Box::<InformationGain>::default();
    let mut builder = DL85Builder::default()
        .max_depth(depth)
        .min_support(1)
        .always_sort(true)
        .specialization(match config.contains("d2") {
            true => OptimalDepth2Policy::Enabled,
            false => OptimalDepth2Policy::Disabled,
        })
        .branching_strategy(match config.contains("dyn") {
            true => BranchingPolicy::Dynamic,
            false => BranchingPolicy::Default,
        })
        .lower_bound_strategy(match config.contains("sim") {
            true => LowerBoundPolicy::Similarity,
            false => LowerBoundPolicy::Disabled,
        });
    let rule: Option<Box<dyn Rule>> = if config.ends_with("lds") {
        Some(Box::new(DiscrepancyRule::new(
            usize::MAX,
            Box::new(Monotonic::new(1)),
        )))
    } else if config.ends_with("topk") {
        Some(Box::new(TopkRule::new(
            cover.num_attributes,
            Box::new(Monotonic::new(1)),
        )))
    } else {
        None
    };
    if let Some(rule) = rule {
        builder = builder.add_search_rule(rule);
    }
    let cache: Box<dyn Caching> = match cache {
        "cover" => Box::<CoverMap>::default(),
        _ => Box::<Trie>::default(),
    };
    let mut algo = builder
        .cache(cache)
        .heuristic(heuristic)
        .depth2_search(Box::new(ErrorMinimizer::new(error_fn.clone())))
        .error_function(error_fn)
        .build()
        .expect("the search builds");
    algo.fit(&mut cover).expect("the search runs");
    let statistics = algo.statistics();
    (
        statistics.tree_error,
        statistics.search_space_size,
        statistics.cache_size,
    )
}

#[test]
fn search_effort_is_unchanged() {
    let mut actual = vec![];
    for dataset in DATASETS {
        for (config, depth) in CONFIGS {
            for cache in CACHES {
                let (error, nodes, entries) = solve(dataset, config, depth, cache);
                actual.push((dataset, config, depth, cache, error, nodes, entries));
            }
        }
    }
    if actual != EXPECTED {
        let mut table = String::new();
        for (dataset, config, depth, cache, error, nodes, entries) in &actual {
            table += &format!(
                "    ({dataset:?}, {config:?}, {depth}, {cache:?}, {error:?}, {nodes}, {entries}),\n"
            );
        }
        panic!("the search changed; the table is now:\n{table}");
    }
}
