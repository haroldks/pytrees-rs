# Rust crates

The Python package is a thin layer over two Rust libraries, which can be used
on their own. They are not published on crates.io yet; depend on them from
the repository:

```toml
[dependencies]
dtrees-rs = { git = "https://github.com/haroldks/pytrees-rs" }
contree-rs = { git = "https://github.com/haroldks/pytrees-rs" }
```

Run `cargo doc --open -p dtrees-rs -p contree-rs` for the full API
documentation.

## dtrees-rs

Decision trees over binary features: DL8.5, the search rules that make it
anytime, LGDT and the depth-2 solvers. Data is loaded into a `Cover`, which
tracks the rows reaching the current node of the search.

```rust
use dtrees_rs::algorithms::greedy::factories::with_error_minimizer;
use dtrees_rs::algorithms::TreeSearchAlgorithm;
use dtrees_rs::reader::data_reader::DataReader;
use std::path::Path;

let mut cover = DataReader::default().read_file(Path::new("data.txt"))?;
let mut lgdt = with_error_minimizer().max_depth(4).min_support(5).build()?;
lgdt.fit(&mut cover)?;
println!("{}", lgdt.tree());
```

DL8.5 is assembled with `DL85Builder`, which takes the cache, the depth-2
solver, the error function and the heuristic as separate parts, and any number
of search rules:

```rust
use dtrees_rs::algorithms::common::errors::NativeError;
use dtrees_rs::algorithms::common::heuristics::InformationGain;
use dtrees_rs::algorithms::optimal::depth2::ErrorMinimizer;
use dtrees_rs::algorithms::optimal::dl85::DL85Builder;
use dtrees_rs::algorithms::optimal::rules::{DiscrepancyRule, Monotonic};
use dtrees_rs::algorithms::TreeSearchAlgorithm;
use dtrees_rs::caching::Trie;

let error_fn = Box::<NativeError>::default();
let mut dl85 = DL85Builder::default()
    .max_depth(4)
    .min_support(5)
    .max_time(60.0)
    .always_sort(true)
    .add_search_rule(Box::new(DiscrepancyRule::new(usize::MAX, Box::<Monotonic>::default())))
    .cache(Box::<Trie>::default())
    .heuristic(Box::<InformationGain>::default())
    .depth2_search(Box::new(ErrorMinimizer::new(error_fn.clone())))
    .error_function(error_fn)
    .build()?;
dl85.fit(&mut cover)?;
```

The `examples/` directory of the crate has one program per search rule.

### Custom error functions

DL8.5 minimises the sum of the errors of the leaves, and the error of a leaf
is whatever the `ErrorWrapper` passed to `error_function` computes. It
receives the class counts of the leaf, or its row ids when the builder is set
to `node_exposed_data(NodeDataType::Tids)`, and returns
`(error, predicted class)`:

```rust
use dtrees_rs::algorithms::common::errors::ErrorWrapper;

/// Misclassification cost that differs per class.
#[derive(Clone)]
struct CostSensitive {
    costs: Vec<f64>,
}

impl ErrorWrapper for CostSensitive {
    fn compute(&self, class_counts: &[usize]) -> (f64, f64) {
        let total: f64 = class_counts.iter().zip(&self.costs).map(|(&n, c)| n as f64 * c).sum();
        // Predict the class whose rows are the costliest to get wrong.
        (0..class_counts.len())
            .map(|k| (total - class_counts[k] as f64 * self.costs[k], k as f64))
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .unwrap_or((0.0, 0.0))
    }
}

let error_fn = Box::new(CostSensitive { costs: vec![1.0, 5.0] });
let mut dl85 = DL85Builder::default()
    .max_depth(3)
    .cache(Box::<Trie>::default())
    .heuristic(Box::<NoHeuristic>::default())
    .depth2_search(Box::new(ErrorMinimizer::new(error_fn.clone())))
    .error_function(error_fn)
    .build()?;
```

A plain function works too, through `NativeError::new`. Two options assume
the misclassification error: the similarity lower bound
(`LowerBoundPolicy::Similarity`) is only valid when each row adds at most 1
to the error, and the depth-2 solver needs class counts, so it is skipped
with row ids.

## contree-rs

Optimal decision trees over continuous features: `ConTree` (exact) and
`ConTreeLds` (anytime).

```rust
use contree::algorithms::ConTree;
use contree::common::{PointSelector, SearchConfig, SearchStatus};
use contree::data::Dataset;

// Row-major values and labels 0..k.
let dataset = Dataset::from_rows(&values, &labels, n_features)?;
let config = SearchConfig::new(1, 3, 600.0, 0, usize::MAX, false, true, PointSelector::Mid);
let outcome = ConTree::with_config(config).fit(&dataset)?;

if outcome.status == SearchStatus::Optimal {
    println!("{} training errors\n{}", outcome.error(), outcome.tree);
}
```

`SearchConfig::new` takes, in order: minimum support, maximum depth, time
limit, tolerated gap, initial error bound, Gini ordering, depth-2 solver and
point selector. `ConTreeLds` is built the same way; `with_schedule` chooses
its budget schedule and `trajectory()` returns every improvement as
`(seconds, error)`.

The library is imported as `contree` (package `contree-rs`).
