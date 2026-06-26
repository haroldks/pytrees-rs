use clap::Parser;
use dtrees_rs::algorithms::common::heuristics::Heuristic;
use dtrees_rs::algorithms::greedy::GreedyBuilder;
use dtrees_rs::algorithms::TreeSearchAlgorithm;
use dtrees_rs::parsers::examples::{load_or_create_result, save_results, ExampleParser, Res};
use dtrees_rs::reader::data_reader::DataReader;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = ExampleParser::parse();
    let method = "greedy";

    if !app.input.exists() {
        return Err(format!("File does not exist: {}", app.input.display()).into());
    }
    let file = app
        .input
        .to_str()
        .ok_or_else(|| format!("Invalid UTF-8 in path: {}", app.input.display()))?;

    let depth = app.depth;
    let support = app.support;
    let time_limit = app.timeout;
    let lambda = app.lambda;

    let path = Path::new(file);
    let file_name = path.file_stem().ok_or("input path has no file stem")?;
    let result_file = app.result.join(file_name);

    let depth_dir = result_file.join(format!("{depth}"));
    fs::create_dir_all(&depth_dir)?;

    let result_path = depth_dir.join(format!("{lambda}.json"));
    let mut result = load_or_create_result(
        &result_path,
        app.overwrite,
        Res {
            lookahead_depth: Some(app.lookahead_depth),
            fast_d2: false,
            ..app.fresh_result(file, method)
        },
    );

    let mut cover = DataReader::default().read_file(path)?;

    let mut algo = GreedyBuilder::default()
        .max_depth(depth)
        .min_support(support)
        .regularization(lambda)
        .heuristic(<Box<dyn Heuristic>>::from(app.heuristic))
        .max_time(time_limit)
        .build()?;

    algo.fit(&mut cover)?;
    result.lambdas.push(algo.tree().root_lambda());
    result.errors.push(algo.tree().root_error());
    result.tree = algo.tree().clone();

    result.completed = true;
    save_results(&result, &result_path)?;

    app.print_tree_if(&result.tree);

    Ok(())
}
