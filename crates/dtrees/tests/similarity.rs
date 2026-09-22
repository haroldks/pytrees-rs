//! DL8.5 with the similarity lower bound and dynamic branching must still
//! find the optimal tree. Each case is checked against a brute-force search
//! over every tree of the given depth.

use dtrees_rs::algorithms::common::errors::NativeError;
use dtrees_rs::algorithms::common::heuristics::NoHeuristic;
use dtrees_rs::algorithms::common::types::{
    BranchingPolicy, LowerBoundPolicy, OptimalDepth2Policy,
};
use dtrees_rs::algorithms::optimal::depth2::ErrorMinimizer;
use dtrees_rs::algorithms::optimal::dl85::DL85Builder;
use dtrees_rs::algorithms::TreeSearchAlgorithm;
use dtrees_rs::bitsets::{BitCollection, Bitset, BitsetInit};
use dtrees_rs::caching::Trie;
use dtrees_rs::cover::Cover;

/// A small deterministic generator, so a failing case can be replayed.
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) % n as u64) as usize
    }
}

struct Instance {
    rows: Vec<Vec<bool>>,
    labels: Vec<usize>,
    num_labels: usize,
}

impl Instance {
    fn cover(&self) -> Cover {
        let n = self.rows.len();
        let num_features = self.rows[0].len();
        let mut attributes = vec![Bitset::new(BitsetInit::Empty(n)); num_features];
        let mut labels = vec![Bitset::new(BitsetInit::Empty(n)); self.num_labels];
        for (row, values) in self.rows.iter().enumerate() {
            for (feature, &value) in values.iter().enumerate() {
                if value {
                    attributes[feature].set(row);
                }
            }
            labels[self.labels[row]].set(row);
        }
        Cover::new(attributes, labels, n)
    }

    /// The fewest errors any tree of at most `depth` levels makes on `rows`.
    fn brute_force(&self, rows: &[usize], depth: usize) -> usize {
        let mut counts = vec![0; self.num_labels];
        for &row in rows {
            counts[self.labels[row]] += 1;
        }
        let leaf = rows.len() - counts.iter().max().copied().unwrap_or(0);
        if depth == 0 || leaf == 0 {
            return leaf;
        }
        let mut best = leaf;
        for feature in 0..self.rows[0].len() {
            let (left, right): (Vec<usize>, Vec<usize>) =
                rows.iter().partition(|&&row| !self.rows[row][feature]);
            if left.is_empty() || right.is_empty() {
                continue;
            }
            best =
                best.min(self.brute_force(&left, depth - 1) + self.brute_force(&right, depth - 1));
        }
        best
    }
}

fn dl85_with_similarity(instance: &Instance, depth: usize) -> f64 {
    let mut cover = instance.cover();
    let error_fn = Box::<NativeError>::default();
    let mut search = DL85Builder::default()
        .max_depth(depth)
        .min_support(1)
        .max_time(60.0)
        .always_sort(true)
        .specialization(OptimalDepth2Policy::Disabled)
        .lower_bound_strategy(LowerBoundPolicy::Similarity)
        .branching_strategy(BranchingPolicy::Dynamic)
        .cache(Box::<Trie>::default())
        .heuristic(Box::<NoHeuristic>::default())
        .depth2_search(Box::new(ErrorMinimizer::new(error_fn.clone())))
        .error_function(error_fn)
        .build()
        .expect("a complete builder");
    search.fit(&mut cover).expect("the search runs");
    search.error()
}

#[test]
fn the_similarity_bound_keeps_the_search_optimal() {
    let mut rng = Lcg(7);
    let mut wrong = Vec::new();
    for case in 0..120 {
        let n = 20 + rng.below(40);
        let num_features = 4 + rng.below(5);
        let num_labels = 2 + rng.below(2);
        let rows: Vec<Vec<bool>> = (0..n)
            .map(|_| (0..num_features).map(|_| rng.below(2) == 1).collect())
            .collect();
        // Mostly the XOR of the first two features, with noise.
        let labels: Vec<usize> = rows
            .iter()
            .map(|r| (usize::from(r[0] ^ r[1]) + usize::from(rng.below(10) < 3)) % num_labels)
            .collect();
        let instance = Instance {
            rows,
            labels,
            num_labels,
        };
        let all: Vec<usize> = (0..n).collect();
        for depth in [2, 3] {
            let expected = instance.brute_force(&all, depth);
            let got = dl85_with_similarity(&instance, depth);
            if got != expected as f64 {
                wrong.push(format!(
                    "case {case}, depth {depth}: {got} errors, optimum {expected}"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "suboptimal trees:\n{}", wrong.join("\n"));
}
