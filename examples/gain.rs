use clap::Parser;
use dtrees_rs::cache::{Caching, Trie};
use dtrees_rs::data::{BinaryData, FileReader};
use dtrees_rs::example::{load_results, save_results, ExampleParser, Res};
use dtrees_rs::heuristics::InformationGain;
use dtrees_rs::searches::errors::NativeError;
use dtrees_rs::searches::optimal::{
    Discrepancy, ExponentialDiscrepancy, LubyDiscrepancy, MonotonicDiscrepancy, RelativeGainDL85,
};
use dtrees_rs::searches::{
    BranchingStrategy, CacheInitStrategy, DiscrepancyStrategy, LowerBoundStrategy, NodeExposedData,
    Specialization,
};
use dtrees_rs::structures::RevBitset;
use std::fs;
use std::fs::remove_file;
use std::path::Path;

fn main() {
    let app = ExampleParser::parse();
    let method = "gain".to_string();

    assert!(app.input.exists(), "File does not exist");

    let file = app.input.to_str().unwrap();
    let depth = app.depth;
    let support = app.support;
    let fast_d2 = app.fast_d2;
    let time_limit = app.timeout;
    let metric = app.metric;
    let epsilon = app.epsilon;
    let one_time_sort = !app.always_sort;

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

    let sub = match app.discrepancy {
        DiscrepancyStrategy::Monotonic => "monotonic",
        DiscrepancyStrategy::Exponential => "exponential",
        DiscrepancyStrategy::Luby => "luby",
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
    let heuristics = Box::<InformationGain>::default();

    let discrepancy: Box<dyn Discrepancy + Send> = match app.discrepancy {
        DiscrepancyStrategy::Monotonic => Box::<MonotonicDiscrepancy>::default(),
        DiscrepancyStrategy::Exponential => Box::<ExponentialDiscrepancy>::default(),
        DiscrepancyStrategy::Luby => Box::<LubyDiscrepancy>::default(),
    };

    let mut learner = RelativeGainDL85::new(
        support,
        depth,
        <f64>::INFINITY,
        metric,
        epsilon,
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
        discrepancy,
        error_function,
        heuristics,
    );
    let mut counter = 0;
    while !learner.time_is_up() {
        let r = learner.partial_fit(&mut structure, None);
        result.errors.push(r.0);
        result.cache.push(learner.cache.size());
        result.metric.push(learner.get_metric());
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
        //break
    }

    result.completed = true;
    result.tree = learner.tree.clone();
    let _ = save_results(&result, &result_path);

    if app.print_stats {
        println!("{:?}", learner.statistics);
    }
}
