use clap::Parser;
use dtrees_rs::algorithms::common::errors::NativeError;
use dtrees_rs::algorithms::common::heuristics::{
    GiniIndex, Heuristic, InformationGain, NoHeuristic,
};
use dtrees_rs::algorithms::common::types::{OptimalDepth2Policy, SearchHeuristic};
use dtrees_rs::algorithms::optimal::depth2::ErrorMinimizer;
use dtrees_rs::algorithms::optimal::dl85::DL85Builder;
use dtrees_rs::algorithms::TreeSearchAlgorithm;
use dtrees_rs::caching::Trie;
use dtrees_rs::parsers::examples::{
    load_split_results, save_split_results, ExampleParser, ResSplit,
};
use dtrees_rs::reader::data_reader::DataReader;
use std::fs;
use std::fs::remove_file;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = ExampleParser::parse();
    let method = "split".to_string();

    assert!(app.input.exists(), "File does not exist");

    let file = app.input.to_str().unwrap();
    let depth = app.depth;
    let lookahead_depth = app.lookahead_depth;
    let support = app.support;
    let time_limit = app.timeout;
    let one_time_sort = !app.always_sort;
    let heuristic_strategy = app.heuristic;
    let lambda = app.lambda;

    let path = Path::new(file);
    let file_name = path.file_stem().expect("Invalid file name");
    let mut result_file = app.result.clone();
    result_file.push(file_name);

    let depth_dir = result_file.join(format!("{depth}"));

    fs::create_dir_all(&depth_dir).unwrap_or_else(|_| {
        panic!(
            "Failed to create result directory: {}",
            result_file.display()
        )
    });

    let result_path = depth_dir.join(format!("{method}_{lookahead_depth}_{lambda}.json"));

    // Try to load previous results
    let mut result = match load_split_results(&result_path) {
        Some(res) if res.completed => {
            if !app.overwrite {
                eprintln!("Computation was already completed. Use different parameters or remove the result file to recompute.");
            } else {
                remove_file(&result_path).expect("Error in removing function");
            }
            ResSplit {
                name: file.to_string(),
                method: method.clone(),
                depth,
                lookahead_depth,
                regularization: lambda,
                support,
                metric: Vec::with_capacity(1),
                runtimes: Vec::with_capacity(1),
                errors: Vec::with_capacity(1),
                lambdas: Vec::with_capacity(1),
                cache: Vec::with_capacity(1),
                completed: false,
                one_time_sort,
                tree: Default::default(),
                fast_d2: true,
            }
        }
        Some(res) => res,
        None => ResSplit {
            name: file.to_string(),
            method: method.clone(),
            depth,
            lookahead_depth,
            regularization: lambda,
            support,
            metric: Vec::with_capacity(1),
            runtimes: Vec::with_capacity(1),
            errors: Vec::with_capacity(1),
            lambdas: Vec::with_capacity(1),
            cache: Vec::with_capacity(1),
            completed: false,
            one_time_sort,
            tree: Default::default(),
            fast_d2: true,
        },
    };

    let reader = DataReader::default();
    let path = Path::new(file);
    let mut cover = reader.read_file(path)?;
    let error_fn = Box::<NativeError>::default();
    let depth2 = Box::new(ErrorMinimizer::new(error_fn.clone()));

    let heuristics: Box<dyn Heuristic> = match heuristic_strategy {
        SearchHeuristic::InformationGain => Box::<InformationGain>::default(),
        SearchHeuristic::GiniIndex => Box::<GiniIndex>::default(),
        SearchHeuristic::NoHeuristic => Box::<NoHeuristic>::default(),
        _ => Box::<NoHeuristic>::default(),
    };

    let mut algo = DL85Builder::default()
        .max_depth(depth)
        .min_support(support)
        .max_time(time_limit)
        .regularization(lambda)
        .lookahead_depth(lookahead_depth, None, 0)
        .always_sort(true)
        .specialization(OptimalDepth2Policy::Disabled)
        .cache(Box::<Trie>::default())
        .heuristic(heuristics)
        .depth2_search(depth2)
        .error_function(error_fn)
        .build()?;

    let r = algo.fit(&mut cover);
    let stats = algo.statistics();
    result.errors.push(stats.tree_error);
    result.cache.push(stats.cache_size);
    result.runtimes.push(stats.duration);
    result.lambdas.push(algo.tree().root_lambda());
    result.tree = algo.tree().clone();

    result.completed = true;
    let _ = save_split_results(&result, &result_path);

    if app.print_stats {
        println!("{:?}", algo.statistics());
    }

    if app.print_tree {
        algo.tree().print()
    }

    Ok(())
}
