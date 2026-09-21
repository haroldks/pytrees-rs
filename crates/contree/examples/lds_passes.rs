//! Diagnostic: what the anytime search does pass by pass.
use contree::algorithms::ConTreeLds;
use contree::common::{PointSelector, ScheduleKind};
use contree::data::view::DataView;
use contree::reader::data_reader::DataReader;
use std::path::Path;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (path, depth, sel, limit) = (
        &a[1],
        a[2].parse().unwrap(),
        &a[3],
        a[4].parse::<f64>().unwrap(),
    );
    let sel = if sel == "first" {
        PointSelector::First
    } else {
        PointSelector::Mid
    };
    let ds = DataReader::default().read_file(Path::new(path)).unwrap();
    // Optional: budget schedule, and a cap on the number of passes (for a
    // deterministic comparison of work done).
    let schedule: ScheduleKind = a
        .get(5)
        .map_or(Ok(ScheduleKind::default()), |x| x.parse())
        .unwrap();
    let max_passes: usize = a.get(6).map_or(usize::MAX, |x| x.parse().unwrap());
    let mut s =
        ConTreeLds::new(1, depth, limit, usize::MAX, sel, 0, true, true).with_schedule(schedule);
    let view = DataView::root(&ds, true);
    let t0 = Instant::now();
    let (mut last_err, mut passes, mut last_print) = (usize::MAX, 0usize, 0.0f64);
    loop {
        let done = s.partial_fit(&view);
        passes += 1;
        let st = *s.statistics();
        let t = t0.elapsed().as_secs_f64();
        if st.error < last_err || t - last_print > 5.0 || done {
            println!("pass {passes:>6}  t={t:7.2}s  error={:>6}  discrepancy_budget={:>3}  cache={:>8}  calls={:>9}",
                     st.error, s.config().budget, st.cache_size, st.general_solver_call + st.specialized_solver_call);
            last_err = last_err.min(st.error);
            last_print = t;
        }
        if done || passes >= max_passes {
            println!(
                "final: pass {passes}  error={}  cache={}  calls={}  hits={}",
                st.error,
                st.cache_size,
                st.general_solver_call + st.specialized_solver_call,
                st.cache_hits
            );
            break;
        }
    }
    println!("status: {}  passes: {passes}", s.status());
}
