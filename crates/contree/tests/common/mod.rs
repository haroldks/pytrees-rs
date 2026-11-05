//! Helpers shared by the integration tests: random instances, and a deep
//! check of a tree against the rows it was learnt from.
//!
//! Each test file compiles this module on its own and uses a different part
//! of it.
#![allow(dead_code)]

use contree::data::Dataset;
use contree::tree::Tree;

/// A small deterministic generator. Seeded explicitly so a failure is
/// reproducible from the reported case number alone.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

pub struct Instance {
    /// Row-major values.
    pub values: Vec<f64>,
    pub labels: Vec<usize>,
    pub n_rows: usize,
    pub n_features: usize,
    pub n_labels: usize,
}

impl Instance {
    pub fn random(rng: &mut Lcg, n_rows: usize, n_features: usize, n_labels: usize) -> Self {
        // Few distinct values per column on purpose: ties are where a split
        // enumeration and a search are most likely to disagree.
        let values = (0..n_rows * n_features)
            .map(|_| rng.below(6) as f64)
            .collect();
        let labels = (0..n_rows).map(|_| rng.below(n_labels)).collect();
        Self {
            values,
            labels,
            n_rows,
            n_features,
            n_labels,
        }
    }

    pub fn dataset(&self) -> Dataset {
        Dataset::from_rows(&self.values, &self.labels, self.n_features).expect("valid instance")
    }

    pub fn value(&self, row: usize, feature: usize) -> f64 {
        self.values[row * self.n_features + feature]
    }
}

/// The rows of a dataset, as an instance.
pub fn instance_from(dataset: &Dataset) -> Instance {
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
    Instance {
        values,
        labels,
        n_rows,
        n_features,
        n_labels: dataset.num_labels(),
    }
}

/// A tree in a form that both a [`Tree`] and a frozen baseline record (JSON)
/// convert to, so both get the same check.
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Leaf {
        label: usize,
        error: f64,
    },
    Split {
        feature: usize,
        split: f64,
        error: f64,
        left: Box<Node>,
        right: Box<Node>,
    },
}

impl Node {
    /// The nodes of `tree` from its root. `None` for an empty tree.
    pub fn from_tree(tree: &Tree) -> Option<Node> {
        fn walk(tree: &Tree, index: usize) -> Node {
            let node = tree
                .get_node(index)
                .expect("child indices point into the arena");
            match node.value.feature {
                Some(feature) if node.left != 0 && node.right != 0 => Node::Split {
                    feature,
                    split: node.value.split.expect("an internal node has a threshold"),
                    error: node.value.error,
                    left: Box::new(walk(tree, node.left)),
                    right: Box::new(walk(tree, node.right)),
                },
                _ => Node::Leaf {
                    label: node.value.label.expect("a leaf has a label"),
                    error: node.value.error,
                },
            }
        }
        (!tree.is_empty()).then(|| walk(tree, tree.get_root_index()))
    }

    /// The tree a baseline record stores under `"tree"`.
    pub fn from_json(value: &serde_json::Value) -> Node {
        let number = |key: &str| {
            value[key]
                .as_f64()
                .unwrap_or_else(|| panic!("{key} in {value}"))
        };
        if let Some(label) = value.get("leaf") {
            Node::Leaf {
                label: label.as_u64().expect("a leaf label") as usize,
                error: number("error"),
            }
        } else {
            Node::Split {
                feature: value["feature"].as_u64().expect("a feature") as usize,
                split: number("split"),
                error: number("error"),
                left: Box::new(Node::from_json(&value["left"])),
                right: Box::new(Node::from_json(&value["right"])),
            }
        }
    }

    /// Where two trees first differ in their tests, thresholds, labels or
    /// errors (beyond `tolerance`), as a path from the root; `None` if they
    /// do not.
    pub fn first_difference(&self, other: &Node, tolerance: f64, path: &str) -> Option<String> {
        match (self, other) {
            (Node::Leaf { label: a, error: e }, Node::Leaf { label: b, error: f }) => (a != b
                || (e - f).abs() > tolerance)
                .then(|| format!("{path}: leaf {a} (error {e}) vs leaf {b} (error {f})")),
            (
                Node::Split {
                    feature: fa,
                    split: sa,
                    error: ea,
                    left: la,
                    right: ra,
                },
                Node::Split {
                    feature: fb,
                    split: sb,
                    error: eb,
                    left: lb,
                    right: rb,
                },
            ) => {
                if fa != fb || sa != sb || (ea - eb).abs() > tolerance {
                    return Some(format!(
                        "{path}: x{fa} <= {sa:?} (error {ea}) vs x{fb} <= {sb:?} (error {eb})"
                    ));
                }
                la.first_difference(lb, tolerance, &format!("{path}L"))
                    .or_else(|| ra.first_difference(rb, tolerance, &format!("{path}R")))
            }
            (a, b) => Some(format!("{path}: {a:?} vs {b:?}")),
        }
    }

    /// Whether two trees have the same tests, thresholds and labels, and
    /// errors equal up to `tolerance`.
    pub fn same_as(&self, other: &Node, tolerance: f64) -> bool {
        match (self, other) {
            (Node::Leaf { label: a, error: e }, Node::Leaf { label: b, error: f }) => {
                a == b && (e - f).abs() <= tolerance
            }
            (
                Node::Split {
                    feature: fa,
                    split: sa,
                    error: ea,
                    left: la,
                    right: ra,
                },
                Node::Split {
                    feature: fb,
                    split: sb,
                    error: eb,
                    left: lb,
                    right: rb,
                },
            ) => {
                fa == fb
                    && sa == sb
                    && (ea - eb).abs() <= tolerance
                    && la.same_as(lb, tolerance)
                    && ra.same_as(rb, tolerance)
            }
            _ => false,
        }
    }
}

/// Checks a tree against the rows it was learnt from, and returns its cost
/// recomputed from them.
///
/// `cost(rows, label)` is what a leaf over `rows` predicting `label` costs.
/// Every node is checked:
/// * a split has a feature of the data, a finite threshold, and rows on both
///   sides (`x <= threshold` goes left); there are at most `max_depth` levels
///   of splits;
/// * a leaf has at least `min_sup` rows, and its label is one of the cheapest
///   for its rows;
/// * the error stored at every node is the cost of its rows, recomputed.
pub fn deep_check(
    node: &Node,
    instance: &Instance,
    max_depth: usize,
    min_sup: usize,
    cost: &dyn Fn(&[usize], usize) -> f64,
) -> Result<f64, String> {
    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs()))
    }
    fn walk(
        node: &Node,
        rows: &[usize],
        depth_left: usize,
        instance: &Instance,
        min_sup: usize,
        cost: &dyn Fn(&[usize], usize) -> f64,
        path: &str,
    ) -> Result<f64, String> {
        match node {
            Node::Leaf { label, error } => {
                if rows.len() < min_sup {
                    return Err(format!(
                        "{path}: leaf of {} rows, min_sup {min_sup}",
                        rows.len()
                    ));
                }
                if *label >= instance.n_labels {
                    return Err(format!(
                        "{path}: label {label} of {} classes",
                        instance.n_labels
                    ));
                }
                let own = cost(rows, *label);
                let best = (0..instance.n_labels)
                    .map(|l| cost(rows, l))
                    .fold(f64::INFINITY, f64::min);
                if !close(own, best) {
                    return Err(format!(
                        "{path}: label {label} costs {own}, the best label {best}"
                    ));
                }
                if !close(own, *error) {
                    return Err(format!("{path}: stores error {error}, its rows cost {own}"));
                }
                Ok(own)
            }
            Node::Split {
                feature,
                split,
                error,
                left,
                right,
            } => {
                if depth_left == 0 {
                    return Err(format!("{path}: a split below the maximum depth"));
                }
                if *feature >= instance.n_features || !split.is_finite() {
                    return Err(format!("{path}: feature {feature}, threshold {split}"));
                }
                let (left_rows, right_rows): (Vec<usize>, Vec<usize>) = rows
                    .iter()
                    .partition(|&&r| instance.value(r, *feature) <= *split);
                if left_rows.is_empty() || right_rows.is_empty() {
                    return Err(format!(
                        "{path}: threshold {split} on feature {feature} leaves a side empty"
                    ));
                }
                let total = walk(
                    left,
                    &left_rows,
                    depth_left - 1,
                    instance,
                    min_sup,
                    cost,
                    &format!("{path}L"),
                )? + walk(
                    right,
                    &right_rows,
                    depth_left - 1,
                    instance,
                    min_sup,
                    cost,
                    &format!("{path}R"),
                )?;
                if !close(total, *error) {
                    return Err(format!(
                        "{path}: stores error {error}, its leaves cost {total}"
                    ));
                }
                Ok(total)
            }
        }
    }
    let rows: Vec<usize> = (0..instance.n_rows).collect();
    walk(node, &rows, max_depth, instance, min_sup, cost, "root ")
}

/// Misclassification of a leaf predicting `label`: the number of its other
/// rows.
pub fn misclassified(instance: &Instance, rows: &[usize], label: usize) -> f64 {
    rows.iter()
        .filter(|&&r| instance.labels[r] != label)
        .count() as f64
}
