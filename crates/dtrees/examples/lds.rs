use clap::Parser;
use dtrees_rs::algorithms::common::errors::NativeError;
use dtrees_rs::algorithms::common::heuristics::{
    GiniIndex, Heuristic, InformationGain, NoHeuristic,
};
use dtrees_rs::algorithms::common::types::{SearchHeuristic, SearchStepStrategy};
use dtrees_rs::algorithms::optimal::depth2::ErrorMinimizer;
use dtrees_rs::algorithms::optimal::dl85::DL85Builder;
use dtrees_rs::algorithms::optimal::rules::{
    DiscrepancyRule, Exponential, Luby, Monotonic, StepStrategy,
};
use dtrees_rs::algorithms::optimal::Reason;
use dtrees_rs::algorithms::TreeSearchAlgorithm;
use dtrees_rs::caching::Trie;
use dtrees_rs::parsers::examples::{load_results, save_results, ExampleParser, Res};
use dtrees_rs::reader::data_reader::DataReader;
use std::fs;
use std::fs::remove_file;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = ExampleParser::parse();
    let method = "lds".to_string();

    assert!(app.input.exists(), "File does not exist");

    let file = app.input.to_str().unwrap();
    let depth = app.depth;
    let support = app.support;
    let fast_d2 = app.fast_d2;
    let time_limit = app.timeout;
    let one_time_sort = !app.always_sort;
    let heuristic_strategy = app.heuristic;
    let lds_strategy = app.step;

    let checkpoint_interval = 10;

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

    let sub = match app.step {
        SearchStepStrategy::Monotonic => "monotonic",
        SearchStepStrategy::Exponential => "exponential",
        SearchStepStrategy::Luby => "luby",
    };

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

    let discrepancy: DiscrepancyRule = match lds_strategy {
        SearchStepStrategy::Monotonic => {
            DiscrepancyRule::new(usize::MAX, Box::<Monotonic>::default())
        }
        SearchStepStrategy::Exponential => {
            DiscrepancyRule::new(usize::MAX, Box::<Exponential>::default())
        }
        SearchStepStrategy::Luby => DiscrepancyRule::new(usize::MAX, Box::<Luby>::default()),
    };

    let mut algo = DL85Builder::default()
        .max_depth(depth)
        .min_support(support)
        .max_time(time_limit)
        .always_sort(true)
        .add_search_rule(Box::new(discrepancy))
        .specialization(fast_d2)
        .cache(Box::<Trie>::default())
        .heuristic(heuristics)
        .depth2_search(depth2)
        .error_function(error_fn)
        .build()?;

    let mut counter = 0;
    while !algo.time_is_exhausted() {
        let r = algo.partial_fit(&mut cover);
        let stats = algo.statistics();
        result.errors.push(stats.tree_error);
        result.cache.push(stats.cache_size);
        result.runtimes.push(stats.duration);
        result.tree = algo.tree().clone();
        if counter > 0 && counter % checkpoint_interval == 0 {
            let _ = save_results(&result, &result_path);
        }
        counter += 1;
        if r.reason == Reason::Done {
            result.completed = true;
            break;
        }
    }

    result.completed = true;
    result.tree = algo.tree().clone();
    let _ = save_results(&result, &result_path);

    if app.print_stats {
        println!("{:?}", algo.statistics());
    }

    if app.print_tree {
        algo.tree().print()
    }

    Ok(())
}
