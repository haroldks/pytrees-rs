//! The tree a search returns must classify its training data with exactly
//! the error the search reports.
//!
//! The search counts its error while exploring, and the tree is rebuilt from
//! the cache afterwards. A flipped routing convention, a wrong leaf label or a
//! subtree grafted at the wrong index would make the two disagree.

use std::path::PathBuf;

use contree::algorithms::{ConTree, ConTreeLds};
use contree::common::PointSelector;
use contree::data::Dataset;
use contree::reader::data_reader::DataReader;
use contree::tree::Tree;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Rebuilds the row-major training matrix from the column-major dataset.
fn rows(dataset: &Dataset) -> (Vec<f64>, Vec<usize>, usize) {
    let n_rows = dataset.count();
    let n_features = dataset.num_features();
    let mut values = vec![0.0; n_rows * n_features];
    let mut labels = vec![0usize; n_rows];

    for feature in 0..n_features {
        for i in 0..n_rows {
            let point = &dataset[feature][i];
            values[point.tid() * n_features + feature] = point.value();
            labels[point.tid()] = point.label() as usize;
        }
    }
    (values, labels, n_features)
}

fn misclassified(tree: &Tree, dataset: &Dataset) -> usize {
    let (values, labels, n_features) = rows(dataset);
    let predicted = tree
        .predict(&values, n_features)
        .expect("the tree must be walkable");
    predicted
        .iter()
        .zip(&labels)
        .filter(|(p, y)| p != y)
        .count()
}

fn load(name: &str) -> Dataset {
    let mut dataset = DataReader::default()
        .read_file(&fixture(name))
        .unwrap_or_else(|err| panic!("{name}: {err}"));
    dataset.sort_features();
    dataset
}

#[test]
fn the_exhaustive_search_tree_reproduces_its_reported_error() {
    // avila_1k (1000x10) only runs at one setting below, to keep debug
    // builds fast.
    for name in ["hepatitis.txt", "iris.txt", "small.txt"] {
        let dataset = load(name);
        for depth in 1..=2 {
            for fast_d2 in [false, true] {
                for heuristic in [false, true] {
                    let mut solver: ConTree = ConTree::new(
                        1,
                        depth,
                        60.0,
                        usize::MAX,
                        PointSelector::Mid,
                        0,
                        heuristic,
                        fast_d2,
                    );
                    solver.fit(&dataset).expect("fit");

                    let label =
                        format!("{name} depth={depth} fast_d2={fast_d2} heuristic={heuristic}");
                    solver
                        .tree
                        .validate()
                        .unwrap_or_else(|err| panic!("{label}: {err}"));
                    assert_eq!(
                        misclassified(&solver.tree, &dataset),
                        solver.statistics().error,
                        "{label}: the tree does not classify the training set the way the \
                         search says it does"
                    );
                }
            }
        }
    }
}

#[test]
fn a_larger_instance_reproduces_its_reported_error() {
    let dataset = load("avila_1k.txt");
    let mut solver = ConTree::new(1, 2, 60.0, usize::MAX, PointSelector::Mid, 0, false, true);
    let outcome = solver.fit(&dataset).expect("fit");

    outcome.tree.validate().expect("valid tree");
    assert_eq!(misclassified(&outcome.tree, &dataset), outcome.error());
}

#[test]
fn the_lds_tree_reproduces_its_reported_error() {
    // hepatitis has 68 features, and LDS without the depth-2 specialization
    // spends minutes on it in a debug build.
    for (name, fast_d2) in [
        ("small.txt", false),
        ("small.txt", true),
        ("iris.txt", false),
        ("iris.txt", true),
        ("hepatitis.txt", true),
    ] {
        let dataset = load(name);
        {
            let mut solver: ConTreeLds = ConTreeLds::new(
                1,
                2,
                60.0,
                usize::MAX,
                PointSelector::Mid,
                0,
                false,
                fast_d2,
            );
            solver.fit(&dataset).expect("fit");

            let tree = solver.get_solution_tree();
            let label = format!("{name} fast_d2={fast_d2}");
            tree.validate()
                .unwrap_or_else(|err| panic!("{label}: {err}"));
            assert_eq!(
                misclassified(&tree, &dataset),
                solver.statistics().error,
                "{label}: the LDS tree does not classify the training set the way the search \
                 says it does"
            );
        }
    }
}

#[test]
fn minimum_support_is_respected_by_every_leaf() {
    // Checked with and without the depth-2 solver, which handles the bottom
    // two levels on its own.
    let dataset = load("avila_1k.txt");
    let (values, _, n_features) = rows(&dataset);

    for min_sup in [1, 10, 50] {
        for fast_d2 in [false, true] {
            let mut solver: ConTree = ConTree::new(
                min_sup,
                2,
                60.0,
                usize::MAX,
                PointSelector::Mid,
                0,
                false,
                fast_d2,
            );
            solver.fit(&dataset).expect("fit");

            let mut per_leaf = vec![0usize; solver.tree.nodes().len()];
            for row in values.chunks_exact(n_features) {
                let path = solver.tree.decision_path(row).expect("walkable");
                per_leaf[*path.last().expect("non-empty path")] += 1;
            }

            for (index, count) in per_leaf.iter().enumerate() {
                if *count == 0 {
                    continue;
                }
                assert!(
                    *count >= min_sup,
                    "min_sup={min_sup} fast_d2={fast_d2}: leaf {index} covers {count} instances"
                );
            }
        }
    }
}

#[test]
fn a_dataset_built_from_memory_matches_one_read_from_a_file() {
    let from_file = load("hepatitis.txt");
    let (values, labels, n_features) = rows(&from_file);
    let from_memory = Dataset::from_rows(&values, &labels, n_features).expect("valid");

    assert_eq!(from_memory.count(), from_file.count());
    assert_eq!(from_memory.num_features(), from_file.num_features());
    assert_eq!(from_memory.num_labels(), from_file.num_labels());

    let mut a: ConTree = ConTree::new(1, 2, 60.0, usize::MAX, PointSelector::Mid, 0, false, true);
    a.fit(&from_file).expect("fit");
    let mut b: ConTree = ConTree::new(1, 2, 60.0, usize::MAX, PointSelector::Mid, 0, false, true);
    b.fit(&from_memory).expect("fit");

    assert_eq!(a.statistics().error, b.statistics().error);
}

#[test]
fn fit_rejects_what_it_used_to_assert_about_in_debug_only() {
    use contree::common::SearchError;

    let prepared = load("iris.txt");
    let n_features = prepared.num_features();
    let n_rows = prepared.count();

    let cases: Vec<(&str, Dataset, usize, f64, SearchError)> = vec![
        (
            "empty dataset",
            Dataset::new(),
            1,
            60.0,
            SearchError::EmptyDataset,
        ),
        (
            "min_sup of zero",
            Dataset::from_rows(
                &vec![0.0; n_rows * n_features],
                &vec![0; n_rows],
                n_features,
            )
            .unwrap(),
            0,
            60.0,
            SearchError::InvalidParameter {
                name: "min_sup",
                reason: "must be at least 1".to_string(),
            },
        ),
    ];

    for (label, dataset, min_sup, max_time, expected) in cases {
        let mut solver = ConTree::new(
            min_sup,
            2,
            max_time,
            usize::MAX,
            PointSelector::Mid,
            0,
            false,
            false,
        );
        assert_eq!(solver.fit(&dataset).unwrap_err(), expected, "{label}");
    }

    // A support larger than half the dataset admits no split at all.
    let mut solver = ConTree::new(
        n_rows,
        2,
        60.0,
        usize::MAX,
        PointSelector::Mid,
        0,
        false,
        false,
    );
    assert!(matches!(
        solver.fit(&prepared).unwrap_err(),
        SearchError::InvalidParameter {
            name: "min_sup",
            ..
        }
    ));

    // A dataset that was never sorted and indexed is refused.
    let mut unprepared = Dataset::new();
    for row in 0..4 {
        for feature in 0..2 {
            unprepared.insert(
                contree::data::DataPoint::new(row, row as f64, (row % 2) as f64),
                feature,
            );
        }
    }
    unprepared.set_num_label(2);
    let mut solver = ConTree::new(1, 2, 60.0, usize::MAX, PointSelector::Mid, 0, false, false);
    assert_eq!(
        solver.fit(&unprepared).unwrap_err(),
        SearchError::UnpreparedDataset
    );
}

#[test]
fn a_fit_reports_why_it_stopped() {
    use contree::common::SearchStatus;

    let dataset = load("iris.txt");

    let mut solver = ConTree::new(1, 2, 60.0, usize::MAX, PointSelector::Mid, 0, false, true);
    assert_eq!(solver.fit(&dataset).unwrap().status, SearchStatus::Optimal);

    // A time limit small enough that the search cannot finish must not be
    // reported as an optimum.
    let mut solver = ConTree::new(1, 4, 1e-9, usize::MAX, PointSelector::Mid, 0, false, false);
    assert_eq!(
        solver.fit(&dataset).unwrap().status,
        SearchStatus::TimeLimit
    );
}
