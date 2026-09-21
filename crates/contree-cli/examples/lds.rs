use clap::Parser;
use contree::algorithms::GenericConTree;
use contree::common::Statistics;
use contree::data::view::DataView;
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::{remove_file, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
#[path = "../src/parsers.rs"]
mod parsers;

use crate::parsers::GeneralParser;
use contree::reader::data_reader::DataReader;
use contree::reader::DataReaderError;
use contree::tree::Tree;

#[derive(Serialize, Deserialize, Clone)]
pub struct Res {
    pub name: String,
    pub depth: usize,
    pub support: usize,
    pub completed: bool,
    pub heuristic: bool,
    pub fast_d2: bool,
    pub runtimes: Vec<f64>,
    pub errors: Vec<usize>,
    pub cache_hits: Vec<usize>,
    pub cache_size: Vec<usize>,
    pub general_solver_calls: Vec<usize>,
    pub specialized_solver_call: Vec<usize>,
    pub trees: Vec<Tree>,
}

pub fn save_results(result: &Res, result_path: &PathBuf) -> std::io::Result<()> {
    // Create parent directories if they don't exist
    if let Some(parent) = result_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let file = File::create(result_path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, result)?;
    writer.flush()
}

pub fn load_results(result_path: &PathBuf) -> Option<Res> {
    if !result_path.exists() {
        return None;
    }

    File::open(result_path).ok().and_then(|file| {
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).ok()
    })
}

pub fn remove_results(result_path: &PathBuf) -> std::io::Result<()> {
    if result_path.exists() {
        remove_file(result_path)?
    }
    Ok(())
}

pub fn push_from_stats(result: &mut Res, stats: &Statistics) {
    result.errors.push(stats.error);
    result.runtimes.push(stats.duration);
    result.cache_size.push(stats.cache_size);
    result.cache_hits.push(stats.cache_hits);
    result.general_solver_calls.push(stats.general_solver_call);
    result
        .specialized_solver_call
        .push(stats.specialized_solver_call);
}

fn main() -> Result<(), DataReaderError> {
    let app = GeneralParser::parse();
    let file = Path::new(&app.input);
    let file_name = file.file_stem().expect("Invalid file name");
    let mut result_file = app.result_dir.clone();
    result_file.push(file_name);

    fs::create_dir_all(&result_file).unwrap_or_else(|_| {
        panic!(
            "Failed to create result directory: {}",
            result_file.display()
        )
    });

    let depth = app.depth;
    let result_path = result_file.join(format!("{depth}.json"));

    let mut result = match load_results(&result_path) {
        Some(res) if res.completed => {
            if !app.overwrite {
                eprintln!("Computation was already completed. Use different parameters or remove the result file to recompute.");
            } else {
                remove_file(&result_path).expect("Error in removing function");
            }
            Res {
                name: file_name.to_str().unwrap().to_string(),
                depth,
                support: app.support,
                runtimes: Vec::with_capacity(100),
                errors: Vec::with_capacity(100),
                cache_hits: vec![],
                cache_size: Vec::with_capacity(100),
                general_solver_calls: vec![],
                completed: false,
                heuristic: app.sort_by_heuristic,
                fast_d2: true,
                specialized_solver_call: vec![],
                trees: vec![],
            }
        }

        Some(res) => res,

        None => Res {
            name: file_name.to_str().unwrap().to_string(),
            depth,
            support: app.support,
            completed: false,
            heuristic: app.sort_by_heuristic,
            fast_d2: true,
            runtimes: vec![],
            errors: vec![],
            cache_hits: vec![],
            cache_size: vec![],
            general_solver_calls: vec![],
            specialized_solver_call: vec![],
            trees: vec![],
        },
    };

    let reader = DataReader::default();
    let mut dataset = reader.read_file(&app.input)?;
    dataset.sort_features();
    let view = DataView::root(&dataset, app.sort_by_heuristic);
    let mut solver: GenericConTree = GenericConTree::from(&app);
    let mut is_done = false;

    let mut counter = 0;
    let checkpoint_interval = 10;
    let mut last = usize::MAX;
    while !is_done {
        is_done = solver.partial_fit(&view).expect("partial fit");
        let stat = solver.stats();
        if stat.error < last {
            result.trees.push(solver.tree());
            last = stat.error;
        }
        push_from_stats(&mut result, &stat);

        if counter > 0 && counter % checkpoint_interval == 0 {
            save_results(&result, &result_path).expect("Failed to save file");
        }
        counter += 1;
    }
    result.completed = true;
    save_results(&result, &result_path).expect("Failed to save file");

    if app.print_stats {
        println!("{:#?}", solver.stats());
    }

    Ok(())
}
