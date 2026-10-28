//! Iterative deepening DL8.5: solves depth 1, then depth 2 with the error of
//! depth 1 as upper bound, and so on up to `--depth`, all within one time
//! limit. Each depth is a fresh DL8.5 restarted every `--restart-interval`
//! seconds, and the best error so far is recorded after every pass as JSON.
//! Run with `--help` for the options.

use clap::Parser;
use dtrees_rs::algorithms::common::errors::NativeError;
use dtrees_rs::algorithms::common::heuristics::{
    GiniIndex, Heuristic, InformationGain, NoHeuristic,
};
use dtrees_rs::algorithms::common::types::SearchHeuristic;
use dtrees_rs::algorithms::optimal::depth2::ErrorMinimizer;
use dtrees_rs::algorithms::optimal::dl85::DL85Builder;
use dtrees_rs::algorithms::optimal::rules::common::TimeLimitRule;
use dtrees_rs::algorithms::optimal::Reason;
use dtrees_rs::algorithms::TreeSearchAlgorithm;
use dtrees_rs::parsers::examples::{save_results, ExampleParser, Res};
use dtrees_rs::reader::data_reader::DataReader;
use std::fs;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = ExampleParser::parse();
    let method = "iterative".to_string();
    let method = app.method_name(method);

    assert!(app.input.exists(), "File does not exist");

    let file = app
        .input
        .to_str()
        .ok_or("the input path is not valid UTF-8")?;
    let time_limit = app.timeout;

    let path = Path::new(file);
    let file_name = path.file_stem().expect("Invalid file name");
    let mut result_file = app.result.clone();
    result_file.push(file_name);

    fs::create_dir_all(&result_file).unwrap_or_else(|_| {
        panic!(
            "Failed to create result directory: {}",
            result_file.display()
        )
    });

    let result_path = result_file.join(format!("{}_{method}.json", app.depth));

    let mut result = Res {
        name: file.to_string(),
        method,
        depth: app.depth,
        support: app.support,
        metric: Vec::with_capacity(100),
        runtimes: Vec::with_capacity(100),
        errors: Vec::with_capacity(100),
        cache: Vec::with_capacity(100),
        completed: false,
        one_time_sort: !app.always_sort,
        tree: Default::default(),
        fast_d2: true,
    };

    let reader = DataReader::default();
    let mut cover = reader.read_file(path)?;

    let start = Instant::now();
    let mut best_error = app.max_error;

    for depth in 1..=app.depth {
        let remaining = time_limit - start.elapsed().as_secs_f64();
        if remaining <= 0.0 {
            break;
        }

        let error_fn = Box::<NativeError>::default();
        let depth2 = Box::new(ErrorMinimizer::new(error_fn.clone()));
        let heuristics: Box<dyn Heuristic> = match app.heuristic {
            SearchHeuristic::InformationGain => Box::<InformationGain>::default(),
            SearchHeuristic::GiniIndex => Box::<GiniIndex>::default(),
            _ => Box::<NoHeuristic>::default(),
        };
        let uninterrupted = app.restart_interval <= 0.0;

        // Bound each depth by the best error of the shallower ones.
        let mut builder = DL85Builder::default()
            .max_depth(depth)
            .min_support(app.support)
            .max_time(remaining)
            .max_error(best_error)
            .always_sort(true);
        if !uninterrupted {
            let time_rule = TimeLimitRule::new(app.restart_interval).relaxable();
            builder = builder.add_search_rule(Box::new(time_rule));
        }
        let depth_start = start.elapsed().as_secs_f64();
        let mut algo = builder
            .specialization(app.fast_d2)
            .cache(app.new_cache())
            .heuristic(heuristics)
            .depth2_search(depth2)
            .error_function(error_fn)
            .build()?;

        // `metric` records the depth being searched.
        let mut finished = false;
        while !algo.time_is_exhausted() {
            let r = algo.partial_fit(&mut cover);
            let stats = algo.statistics();
            let previous_best = best_error;
            if uninterrupted {
                // Record each improvement, timed from the start of the run.
                for &(time, incumbent) in algo.incumbents() {
                    if incumbent < best_error {
                        best_error = incumbent;
                        result.errors.push(incumbent);
                        result.cache.push(stats.cache_size);
                        result.runtimes.push(depth_start + time);
                        result.metric.push(depth as f64);
                    }
                }
            }
            if stats.tree_error < previous_best {
                best_error = best_error.min(stats.tree_error);
                result.tree = algo.tree().clone();
            }
            if best_error.is_finite() {
                result.errors.push(best_error);
                result.cache.push(stats.cache_size);
                result.runtimes.push(start.elapsed().as_secs_f64());
                result.metric.push(depth as f64);
            }
            if r.reason == Reason::Done {
                finished = true;
                break;
            }
        }

        let _ = save_results(&result, &result_path);
        if !finished || best_error <= 0.0 {
            break;
        }
    }

    result.completed = true;
    let _ = save_results(&result, &result_path);

    if app.print_stats {
        println!("best error: {best_error}");
    }

    if app.print_tree {
        println!("{}", result.tree)
    }

    Ok(())
}
