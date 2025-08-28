use dtrees_rs::cache::{Caching, Trie};
use dtrees_rs::data::BinaryData;
use dtrees_rs::data::FileReader;
use dtrees_rs::heuristics::{
    GiniIndex, Heuristic, InformationGain, InformationGainRatio, NoHeuristic,
};
use dtrees_rs::searches::errors::NativeError;
use dtrees_rs::searches::optimal::{PurityDL85, RestartDL85};
use dtrees_rs::searches::{
    BranchingStrategy, CacheInitStrategy, LowerBoundStrategy, NodeExposedData, SearchHeuristic,
    Specialization,
};
use dtrees_rs::structures::RevBitset;
use std::fs;
use std::fs::remove_file;
use std::path::Path;

use clap::Parser;
use dtrees_rs::example::{load_results, save_results, ExampleParser, Res};

fn main() {
    let app = ExampleParser::parse();
    let method = "restart".to_string();

    assert!(app.input.exists(), "File does not exist");

    let file = app.input.to_str().unwrap();
    let depth = app.depth;
    let support = app.support;
    let fast_d2 = app.fast_d2;
    let time_limit = app.timeout;
    let one_time_sort = !app.always_sort;
    let heuristic_strategy = app.heuristic;

    // How often to save checkpoints (every N iterations)
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

    let result_path = result_file.join(format!("{depth}_{method}.json"));

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
                fast_d2,
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
            fast_d2,
        },
    };

    let data = BinaryData::read(file, false, 0.0);
    let mut structure = RevBitset::new(&data);
    let error_function = Box::<NativeError>::default();
    let cache = Box::<Trie>::default();
    let heuristics: Box<dyn Heuristic + Send> = match heuristic_strategy {
        SearchHeuristic::InformationGain => Box::<InformationGain>::default(),
        SearchHeuristic::InformationGainRatio => Box::<InformationGainRatio>::default(),
        SearchHeuristic::GiniIndex => Box::<GiniIndex>::default(),
        SearchHeuristic::None_ => Box::<NoHeuristic>::default(),
    };

    let mut learner = RestartDL85::new(
        support,
        depth,
        <f64>::INFINITY,
        Some(1),
        time_limit,
        one_time_sort,
        0,
        CacheInitStrategy::None_,
        if fast_d2 {
            Specialization::Murtree
        } else {
            Specialization::None_
        },
        LowerBoundStrategy::None_,
        BranchingStrategy::None_,
        NodeExposedData::ClassesSupport,
        cache,
        error_function,
        heuristics,
    );
    let mut counter = 0;
    while !learner.time_is_up() {
        let r = learner.partial_fit(&mut structure, None);
        result.errors.push(r.0);
        result.cache.push(learner.cache.size());
        result.metric.push(r.1);
        result.runtimes.push(learner.current_runtime());
        result.tree = learner.tree.clone();
        // Save checkpoint at regular intervals
        if counter > 0 && counter % checkpoint_interval == 0 {
            let _ = save_results(&result, &result_path);
        }
        counter += 1;
        if learner.is_optimal() {
            result.completed = true;
            break;
        }
    }

    result.completed = true;
    result.tree = learner.tree.clone();
    let _ = save_results(&result, &result_path);

    // learner.fit(&mut structure);
    if app.print_stats {
        println!("{:?}", learner.statistics);
    }
}
