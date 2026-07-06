use clap::Parser;
use dtrees_rs::algorithms::common::types::SearchHeuristic;
use dtrees_rs::mid::schedule::ThresholdSchedule;
use dtrees_rs::mid::search::{DL85Config, MIDSearch};
use dtrees_rs::mid::ImpurityMetric;
use dtrees_rs::parsers::examples::{load_results, save_results, ExampleParser, Res};
use dtrees_rs::reader::ContinuousDataReader;
use std::fs;
use std::fs::remove_file;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = ExampleParser::parse();
    let method = "mid".to_string();

    assert!(app.input.exists(), "File does not exist");

    let file = app.input.to_str().unwrap();
    let depth = app.depth;
    let support = app.support;
    let time_limit = app.timeout;
    let one_time_sort = !app.always_sort;
    let heuristic_strategy = app.heuristic;

    let checkpoint_interval = 10;

    // Impurity metric used to rank MID thresholds, derived from the heuristic flag.
    let metric = match heuristic_strategy {
        SearchHeuristic::GiniIndex => ImpurityMetric::Gini,
        _ => ImpurityMetric::Entropy,
    };

    let sub = match metric {
        ImpurityMetric::Gini => "gini",
        ImpurityMetric::Entropy => "entropy",
    };

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

    let result_path = result_file.join(format!("{depth}_{method}-{sub}.json"));

    // Try to load previous results
    let mut result = match load_results(&result_path) {
        Some(res) if res.completed => {
            if !app.overwrite {
                eprintln!("Computation was already completed. Use different parameters or remove the result file to recompute.");
            } else {
                remove_file(&result_path).expect("Error in removing function");
            }
            Res {
                name: file.to_string(),
                method: method.clone(),
                depth,
                support,
                metric: Vec::with_capacity(100),
                runtimes: Vec::with_capacity(100),
                errors: Vec::with_capacity(100),
                cache: Vec::with_capacity(100),
                completed: false,
                one_time_sort,
                tree: Default::default(),
                fast_d2: true,
            }
        }
        Some(res) => res,
        None => Res {
            name: file.to_string(),
            method: method.clone(),
            depth,
            support,
            metric: Vec::with_capacity(100),
            runtimes: Vec::with_capacity(100),
            errors: Vec::with_capacity(100),
            cache: Vec::with_capacity(100),
            completed: false,
            one_time_sort,
            tree: Default::default(),
            fast_d2: true,
        },
    };

    let reader = ContinuousDataReader::default();
    let data = reader.read_file(path)?;

    // Anytime schedule: start with 1 binary feature and add 1 at every restart,
    // up to however many thresholds MID computes (limit = 0 means "all").
    let schedule = ThresholdSchedule::new(1, 1, 0);
    let config = DL85Config {
        max_depth: depth,
        min_sup: support,
        max_error: f64::INFINITY,
    };

    let mut algo = MIDSearch::new(&data, metric, config, schedule, time_limit);

    let mut counter = 0;
    loop {
        let more = algo.partial_fit();

        // `metric` tracks the number of MID iterations (binary features) used so far.
        result.metric.push(algo.iteration as f64);
        result.errors.push(algo.error());
        result.cache.push(algo.discretizer().len());
        result.runtimes.push(algo.elapsed());
        result.tree = algo.tree().clone();

        if counter > 0 && counter % checkpoint_interval == 0 {
            let _ = save_results(&result, &result_path);
        }
        counter += 1;

        if !more {
            break;
        }
    }

    result.completed = true;
    result.tree = algo.tree().clone();
    let _ = save_results(&result, &result_path);

    if app.print_stats {
        println!(
            "iterations: {}, best error: {}, runtime: {:.3}s",
            algo.iteration,
            algo.error(),
            algo.elapsed()
        );
    }

    if app.print_tree {
        algo.tree().print()
    }

    Ok(())
}