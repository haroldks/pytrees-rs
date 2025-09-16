use crate::algorithms::common::config::BaseSearchConfig;
use crate::algorithms::common::types::FitError;
use crate::algorithms::optimal::depth2::OptimalDepth2Tree;
use crate::algorithms::TreeSearchAlgorithm;
use crate::cover::Cover;
use crate::globals::{float_is_null, item};
use crate::tree::Tree;

pub mod builder;
mod config;
pub mod factories;

pub struct LGDT<D>
where
    D: OptimalDepth2Tree,
{
    search: Box<D>,
    config: BaseSearchConfig,
    tree: Tree,
}

impl<D> TreeSearchAlgorithm for LGDT<D>
where
    D: OptimalDepth2Tree,
{
    fn fit(&mut self, cover: &mut Cover) -> Result<(), FitError> {
        if self.config.max_depth <= 2 {
            self.tree =
                self.search
                    .fit(self.config.min_support, self.config.max_depth, cover, None)?;
            self.tree.print();
            return Ok(());
        }

        let mut solution_tree = Tree::new();
        let root_index = solution_tree.add_default_root();

        let root_tree = self.search.fit(self.config.min_support, 2, cover, None)?;

        let root_attribute = root_tree.root_test().ok_or(FitError::EmptyTree)?;
        solution_tree
            .update_root()
            .map(|updater| updater.value(root_tree.root_details()));
        self.recursion(
            self.config.max_depth - 1,
            cover,
            &mut solution_tree,
            root_index,
            root_attribute,
        )?;
        self.tree = solution_tree;
        Ok(())
    }

    fn tree(&self) -> &Tree {
        &self.tree
    }
}

impl<D> LGDT<D>
where
    D: OptimalDepth2Tree,
{
    fn recursion(
        &self,
        depth: usize,
        cover: &mut Cover,
        tree: &mut Tree,
        parent: usize,
        attribute: usize,
    ) -> Result<f64, FitError> {
        let mut parent_error = 0.0;
        for branch_value in [0, 1] {
            let support = cover.branch_on(item(attribute, branch_value));

            if support < self.config.min_support {
                parent_error +=
                    self.create_leaf_node_in_tree(tree, parent, branch_value == 0, cover);
                cover.backtrack();
                continue;
            }

            if depth <= 1 {
                let child_tree_result =
                    self.search.fit(self.config.min_support, depth, cover, None);
                parent_error += match child_tree_result {
                    Err(FitError::EmptyTree) | Err(FitError::EmptyCandidates)  => {
                        self.create_leaf_node_in_tree(tree, parent, branch_value == 0, cover)
                    }
                    Ok(child_tree) => {
                        let child_index = tree.create_child(parent, branch_value == 0);
                        tree.update_subtree(child_index, &child_tree, child_tree.get_root_index());
                        child_tree.root_error()
                    }
                    Err(err) => return Err(err),
                };
            } else {
                let child_tree_result = self.search.fit(self.config.min_support, 2, cover, None);
                let child_error_result = match child_tree_result {
                    Err(FitError::EmptyTree) | Err(FitError::EmptyCandidates) => {
                        Ok(self.create_leaf_node_in_tree(tree, parent, branch_value == 0, cover))
                    }
                    Ok(child_tree) => {
                        let mut error = Ok(child_tree.root_error());
                        let child_index = tree.create_child(parent, branch_value == 0);
                        if float_is_null(child_tree.root_error()) {
                            tree.update_subtree(
                                child_index,
                                &child_tree,
                                child_tree.get_root_index(),
                            );
                        } else {
                            tree.update_node(child_index)
                                .map(|updater| updater.value(child_tree.root_details()));
                            let next_attribute = child_tree
                                .node_test(child_tree.get_root_index())
                                .ok_or(FitError::AlgorithmError)?;

                            error =
                                self.recursion(depth - 1, cover, tree, child_index, next_attribute);
                        }

                        error
                    }
                    Err(err) => Err(err),
                };

                parent_error += child_error_result?;
            }
            cover.backtrack();
        }

        tree.update_node(parent)
            .map(|updater| updater.error(parent_error));
        Ok(parent_error)
    }

    fn create_leaf_node_in_tree(
        &self,
        tree: &mut Tree,
        parent: usize,
        left: bool,
        cover: &mut Cover,
    ) -> f64 {
        let child_index = tree.create_child(parent, left);
        let error = self.search.error(&cover.labels_count());
        tree.update_leaf_node(child_index, error);
        error.0
    }

    pub fn config(&self) -> &BaseSearchConfig {
        &self.config
    }

}

mod test_lgdt {
    use crate::algorithms::greedy::lgdt::factories::with_error_minimizer;
    use crate::algorithms::TreeSearchAlgorithm;
    use crate::reader::data_reader::DataReader;
    use std::path::Path;

    #[test]
    fn test_d2_lgdt() {
        let reader = DataReader::default();
        let path = Path::new("test_data/anneal.txt");
        let cover_result = reader.read_file(path);

        let mut cover = match cover_result {
            Ok(cover) => cover,
            Err(_) => panic!("oops"),
        };

        let mut lgdt = with_error_minimizer()
            .min_support(1)
            .max_depth(12)
            .build()
            .unwrap();
        let x = lgdt.fit(&mut cover).unwrap();
        lgdt.tree.print()
    }
}
