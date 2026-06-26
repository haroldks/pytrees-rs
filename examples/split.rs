use clap::Parser;
use dtrees_rs::algorithms::common::errors::NativeError;
use dtrees_rs::algorithms::common::heuristics::Heuristic;
use dtrees_rs::algorithms::common::types::CacheType;
use dtrees_rs::algorithms::optimal::depth2::ErrorMinimizer;
use dtrees_rs::algorithms::optimal::dl85::{DL85Builder, HashDL85Builder};
use dtrees_rs::algorithms::TreeSearchAlgorithm;
use dtrees_rs::caching::Trie;
use dtrees_rs::parsers::examples::{load_or_create_result, save_results, ExampleParser, Res};
use dtrees_rs::reader::data_reader::DataReader;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = ExampleParser::parse();
    let method = match app.cache_type {
        CacheType::Trie => "triesplit",
        CacheType::Hashmap => "hashsplit",
    };

    if !app.input.exists() {
        return Err(format!("File does not exist: {}", app.input.display()).into());
    }
    let file = app
        .input
        .to_str()
        .ok_or_else(|| format!("Invalid UTF-8 in path: {}", app.input.display()))?;

    let depth = app.depth;
    let support = app.support;
    let fast_d2 = app.fast_d2;
    let time_limit = app.timeout;
    let lambda = app.lambda;

    let path = Path::new(file);
    let file_name = path.file_stem().ok_or("input path has no file stem")?;
    let result_file = app.result.join(file_name);

    let depth_dir = result_file.join(format!("{depth}"));
    fs::create_dir_all(&depth_dir)?;

    let result_path = depth_dir.join(format!("{method}_{}_{lambda}.json", app.lookahead_depth));
    let mut result = load_or_create_result(
        &result_path,
        app.overwrite,
        Res {
            lookahead_depth: Some(app.lookahead_depth),
            ..app.fresh_result(file, method)
        },
    );

    let mut cover = DataReader::default().read_file(path)?;
    let error_fn = Box::<NativeError>::default();
    let depth2 = Box::new(ErrorMinimizer::new(error_fn.clone()));

    let (stats, tree) = match app.cache_type {
        CacheType::Trie => {
            let mut algo = DL85Builder::default()
                .max_depth(depth)
                .min_support(support)
                .max_time(time_limit)
                .regularization(lambda)
                .lookahead_depth(app.lookahead_depth, None, 0)
                .always_sort(app.always_sort)
                .specialization(fast_d2)
                .cache(Box::<Trie>::default())
                .heuristic(<Box<dyn Heuristic>>::from(app.heuristic))
                .depth2_search(depth2)
                .error_function(error_fn)
                .build()?;
            algo.fit(&mut cover)?;
            (*algo.statistics(), algo.tree().clone())
        }
        CacheType::Hashmap => {
            let mut algo = HashDL85Builder::default()
                .max_depth(depth)
                .min_support(support)
                .max_time(time_limit)
                .regularization(lambda)
                .lookahead_depth(app.lookahead_depth, None, 0)
                .always_sort(app.always_sort)
                .specialization(fast_d2)
                .heuristic(<Box<dyn Heuristic>>::from(app.heuristic))
                .depth2_search(depth2)
                .error_function(error_fn)
                .build()?;
            algo.fit(&mut cover)?;
            (*algo.statistics(), algo.tree().clone())
        }
    };

    result.errors.push(stats.tree_error);
    result.cache.push(stats.cache_size);
    result.runtimes.push(stats.duration);
    result.lambdas.push(tree.root_lambda());
    result.tree = tree;
    result.completed = true;
    save_results(&result, &result_path)?;

    app.print_outcome(stats, &result.tree);

    Ok(())
}
