mod builder;
mod config;

pub use builder::GreedyBuilder;

use crate::algorithms::common::errors::{ErrorWrapper, NativeError};
use crate::algorithms::common::heuristics::Heuristic;
use crate::algorithms::common::types::FitError;
use crate::algorithms::greedy::classic::config::GreedyConfig;
use crate::algorithms::TreeSearchAlgorithm;
use crate::cover::Cover;
use crate::globals::item;
use crate::tree::Tree;

pub struct Greedy<H: Heuristic + ?Sized> {
    config: GreedyConfig,
    error_fn: Box<NativeError>,
    heuristic_fn: Box<H>,
    tree: Tree,
}

impl<H: Heuristic + ?Sized> Greedy<H> {
    pub fn new(config: GreedyConfig, heuristic_fn: Box<H>) -> Self {
        Self {
            config,
            error_fn: Box::<NativeError>::default(),
            heuristic_fn,
            tree: Tree::default(),
        }
    }

    fn recursion(
        &mut self,
        cover: &mut Cover,
        depth: usize,
        tree: &mut Tree,
        index: usize,
    ) -> (f64, f64) {
        if depth == 0 {
            tree.update_node(index).map(|u| u.leaf());
            return (tree.node_error(index), tree.node_lambda(index));
        }

        let mut candidates = self.get_candidates(cover, self.config.base.min_support, None, None);
        if candidates.is_empty() {
            return (tree.node_error(index), tree.node_lambda(index));
        }

        self.heuristic_fn.compute(cover, &mut candidates);
        let best_candidate = candidates[0];

        let (total_error, total_lambda) = [0, 1].iter().fold((0.0, 0.0), |acc, &branch| {
            cover.branch_on(item(best_candidate, branch));

            let error = self.error_fn.compute(&cover.labels_count());
            let child_idx = tree.create_child(index, branch == 0);

            tree.update_node(child_idx)
                .map(|u| u.error(error.0).output(error.1).lambda(self.config.lambda));

            let (child_error, child_lambda) = self.recursion(cover, depth - 1, tree, child_idx);
            cover.backtrack();

            (acc.0 + child_error, acc.1 + child_lambda)
        });

        let split_is_beneficial =
            total_error + total_lambda < tree.node_error(index) + tree.node_lambda(index);

        if split_is_beneficial {
            tree.update_node(index).map(|u| {
                u.error(total_error)
                    .lambda(total_lambda)
                    .test(best_candidate)
            });
            (total_error, total_lambda)
        } else {
            tree.update_node(index).map(|u| u.leaf());
            (tree.node_error(index), tree.node_lambda(index))
        }
    }
}

impl<H: Heuristic + ?Sized> TreeSearchAlgorithm for Greedy<H> {
    fn fit(&mut self, cover: &mut Cover) -> Result<(), FitError> {
        let error = self.error_fn.compute(&cover.labels_count());

        let mut tree = Tree::new();
        let root_index = tree.add_default_root();

        tree.update_root()
            .map(|u| u.lambda(self.config.lambda).error(error.0).output(error.1));

        self.recursion(cover, self.config.base.max_depth, &mut tree, root_index);
        self.tree = tree;

        Ok(())
    }

    fn tree(&self) -> &Tree {
        &self.tree
    }
}

#[cfg(test)]
mod test_greedy {
    use crate::algorithms::common::errors::NativeError;
    use crate::algorithms::common::heuristics::InformationGain;
    use crate::algorithms::greedy::classic::builder::GreedyBuilder;
    use crate::algorithms::TreeSearchAlgorithm;
    use crate::reader::data_reader::DataReader;
    use std::path::Path;

    #[test]
    fn simple_test() -> Result<(), Box<dyn std::error::Error>> {
        let path = Path::new("test_data/anneal.txt");
        let mut cover = DataReader::default()
            .read_file(path)
            .expect("Failed to read test data");

        let mut greedy = GreedyBuilder::default()
            .max_depth(4)
            .min_support(1)
            .regularization(1.0)
            .heuristic(Box::<InformationGain>::default())
            .build()?;

        greedy.fit(&mut cover)?;
        greedy.tree.print();

        Ok(())
    }
}
