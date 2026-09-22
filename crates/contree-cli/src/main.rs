//! `con-tree`: learns an optimal decision tree on continuous features from a
//! text dataset and prints it.
//!
//! ```text
//! con-tree --input data.txt --depth 3 --print-tree --print-stats --result-dir .
//! ```

mod parsers;

use std::process::ExitCode;

use clap::Parser;
use contree::algorithms::GenericConTree;
use contree::reader::data_reader::DataReader;

use crate::parsers::GeneralParser;

fn main() -> ExitCode {
    let app = GeneralParser::parse();

    let reader = DataReader::default();
    let dataset = match reader.read_file(&app.input) {
        Ok(dataset) => dataset,
        Err(err) => {
            eprintln!("con-tree: {}: {err}", app.input.display());
            return ExitCode::FAILURE;
        }
    };

    let mut solver = GenericConTree::from(&app);
    let outcome = match solver.fit(&dataset) {
        Ok(outcome) => outcome,
        Err(err) => {
            eprintln!("con-tree: {err}");
            return ExitCode::FAILURE;
        }
    };

    if app.print_tree {
        println!("{}", outcome.tree);
    }

    if app.print_stats {
        println!("{:#?}", outcome.statistics);
        println!("status: {}", outcome.status);
    }

    ExitCode::SUCCESS
}
