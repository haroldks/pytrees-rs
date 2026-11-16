//! Records the anytime profile of one search: every improvement of the root
//! incumbent as `(seconds, error)`, as JSON on stdout.
//!
//!     anytime <dataset.txt> <depth> <method> <time-limit> [budget-schedule]
//!
//! Methods, all with the depth-2 solver:
//!
//! - `contree` (no ordering) and `contree-gini`: the exhaustive search;
//! - `lds-<selector>-gini`: the anytime search;
//! - `split-<selector>-gini`: anytime SPLIT without the LDS budgets, the
//!   lookahead starting at 0 and one level deeper per pass;
//! - `splitlds-<selector>-gini-<sync|nosync>`: anytime SPLIT with the LDS
//!   budgets, the lookahead deepening synced with the schedule or at every
//!   pass.
//!
//! `<selector>` is `first`, `mid` or `random` (seeded with 0). Without a
//! budget schedule, the one that suits the selector is used.

use std::env;
use std::path::Path;

use contree::algorithms::{ConTree, ConTreeLds};
use contree::common::{Deepening, PointSelector, ScheduleKind};
use contree::reader::data_reader::DataReader;

fn usage() -> ! {
    eprintln!("usage: anytime <dataset.txt> <depth> <method> <time-limit> [budget-schedule]");
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let (path, depth, method, limit, schedule): (_, _, _, _, Option<ScheduleKind>) =
        match args.as_slice() {
            [_, p, d, m, l] => (p, d, m, l, None),
            [_, p, d, m, l, s] => (p, d, m, l, Some(s.parse().expect("budget schedule"))),
            _ => usage(),
        };
    let depth: usize = depth.parse().expect("depth");
    let limit: f64 = limit.parse().expect("time limit");
    let dataset = DataReader::default()
        .read_file(Path::new(path))
        .expect("readable dataset");

    let parts: Vec<&str> = method.split('-').collect();
    let (trajectory, error, status) = match parts.as_slice() {
        ["contree"] | ["contree", "gini"] => {
            let gini = parts.len() == 2;
            let mut s = ConTree::new(
                1,
                depth,
                limit,
                f64::INFINITY,
                PointSelector::Mid,
                0.0,
                gini,
                true,
            );
            let out = s.fit(&dataset).expect("fit");
            (s.trajectory().to_vec(), out.error(), out.status)
        }
        [kind @ ("lds" | "split" | "splitlds"), selector, "gini", rest @ ..] => {
            let selector: PointSelector = selector.parse().unwrap_or_else(|_| usage());
            let mut s = ConTreeLds::new(1, depth, limit, f64::INFINITY, selector, 0.0, true, true)
                .with_random_state(0);
            if let Some(schedule) = schedule {
                s = s.with_schedule(schedule);
            }
            s = match (*kind, rest) {
                ("lds", []) => s,
                ("split", []) => s.with_lds(false).with_lookahead(0, Deepening::EveryPass),
                ("splitlds", ["sync"]) => s.with_lookahead(0, Deepening::Synced),
                ("splitlds", ["nosync"]) => s.with_lookahead(0, Deepening::EveryPass),
                _ => usage(),
            };
            let out = s.fit(&dataset).expect("fit");
            (s.trajectory().to_vec(), out.error(), out.status)
        }
        _ => {
            eprintln!("unknown method {method}");
            usage();
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
