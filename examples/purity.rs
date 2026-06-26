use clap::Parser;
use dtrees_rs::algorithms::common::errors::NativeError;
use dtrees_rs::algorithms::common::heuristics::Heuristic;
use dtrees_rs::algorithms::common::types::CacheType;
use dtrees_rs::algorithms::optimal::depth2::ErrorMinimizer;
use dtrees_rs::algorithms::optimal::dl85::{DL85Builder, HashDL85Builder};
use dtrees_rs::algorithms::optimal::rules::PurityRule;
use dtrees_rs::caching::Trie;
use dtrees_rs::parsers::examples::{load_or_create_result, run_iterative, ExampleParser};
use dtrees_rs::reader::data_reader::DataReader;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = ExampleParser::parse();
    let method = match app.cache_type {
        CacheType::Trie => "triepurity",
        CacheType::Hashmap => "hashpurity",
    };
    let checkpoint = 10;

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
    let result_dir = app.result.join(file_name);

    fs::create_dir_all(&result_dir)?;

    let result_path = result_dir.join(format!("{depth}_{method}_{}.json", app.epsilon));
    let mut result =
        load_or_create_result(&result_path, app.overwrite, app.fresh_result(file, method));

    let mut cover = DataReader::default().read_file(path)?;
    let error_fn = Box::<NativeError>::default();
    let depth2 = Box::new(ErrorMinimizer::new(error_fn.clone()));
    let purity_rule = PurityRule::new(0.0, app.epsilon);

    let stats = match app.cache_type {
        CacheType::Trie => {
            let mut algo = DL85Builder::default()
                .max_depth(depth)
                .min_support(support)
                .max_time(time_limit)
                .always_sort(app.always_sort)
                .regularization(lambda)
                .add_node_rule(Box::new(purity_rule))
                .specialization(fast_d2)
                .cache(Box::<Trie>::default())
                .heuristic(<Box<dyn Heuristic>>::from(app.heuristic))
                .depth2_search(depth2)
                .error_function(error_fn)
                .build()?;
            run_iterative(&mut algo, &mut cover, &mut result, &result_path, checkpoint)?
        }
        CacheType::Hashmap => {
            let mut algo = HashDL85Builder::default()
                .max_depth(depth)
                .min_support(support)
                .max_time(time_limit)
                .always_sort(app.always_sort)
                .regularization(lambda)
                .add_node_rule(Box::new(purity_rule))
                .specialization(fast_d2)
                .heuristic(<Box<dyn Heuristic>>::from(app.heuristic))
                .depth2_search(depth2)
                .error_function(error_fn)
                .build()?;
            run_iterative(&mut algo, &mut cover, &mut result, &result_path, checkpoint)?
        }
    };

    app.print_outcome(stats, &result.tree);

    Ok(())
}
