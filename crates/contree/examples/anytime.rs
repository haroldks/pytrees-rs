//! Records the anytime profile of one search: every improvement of the root
//! incumbent as `(seconds, error)`, as JSON on stdout.
//!
//!     anytime <dataset.txt> <depth> <method> <time-limit> [budget-schedule]
//!
//! Methods: `contree` (no ordering), `contree-gini`, `lds-first-gini` and
//! `lds-mid-gini`. All use the depth-2 solver.

use std::env;
use std::path::Path;

use contree::algorithms::{ConTree, ConTreeLds};
use contree::common::{PointSelector, ScheduleKind};
use contree::reader::data_reader::DataReader;

fn main() {
    let args: Vec<String> = env::args().collect();
    let (path, depth, method, limit, schedule) = match args.as_slice() {
        [_, p, d, m, l] => (p, d, m, l, ScheduleKind::default()),
        [_, p, d, m, l, s] => (p, d, m, l, s.parse().expect("budget schedule")),
        _ => {
            eprintln!(
                "usage: anytime <dataset.txt> <depth> <method> <time-limit> [budget-schedule]"
            );
            std::process::exit(2);
        }
    };
    let depth: usize = depth.parse().expect("depth");
    let limit: f64 = limit.parse().expect("time limit");
    let dataset = DataReader::default()
        .read_file(Path::new(path))
        .expect("readable dataset");

    let (trajectory, error, status) = match method.as_str() {
        "contree" | "contree-gini" => {
            let gini = method == "contree-gini";
            let mut s = ConTree::new(
                1,
                depth,
                limit,
                usize::MAX,
                PointSelector::Mid,
                0,
                gini,
                true,
            );
            let out = s.fit(&dataset).expect("fit");
            (s.trajectory().to_vec(), out.error(), out.status)
        }
        "lds-first-gini" | "lds-mid-gini" => {
            let selector = if method == "lds-first-gini" {
                PointSelector::First
            } else {
                PointSelector::Mid
            };
            let mut s = ConTreeLds::new(1, depth, limit, usize::MAX, selector, 0, true, true)
                .with_schedule(schedule);
            let out = s.fit(&dataset).expect("fit");
            (s.trajectory().to_vec(), out.error(), out.status)
        }
        other => {
            eprintln!("unknown method {other}");
            std::process::exit(2);
        }
    };

    let points: Vec<String> = trajectory
        .iter()
        .map(|(t, e)| format!("[{t:.6},{e}]"))
        .collect();
    println!(
        "{{\"method\":\"{method}\",\"depth\":{depth},\"limit\":{limit},\"error\":{error},\
         \"status\":\"{status}\",\"trajectory\":[{}]}}",
        points.join(",")
    );
}
