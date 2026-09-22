use crate::algorithms::common::config::BaseSearchConfig;
use crate::algorithms::common::types::FitError;
use crate::algorithms::optimal::depth2::OptimalDepth2Tree;
use crate::algorithms::TreeSearchAlgorithm;
use crate::cover::Cover;
use crate::globals::{float_is_null, item};
use crate::tree::Tree;

pub mod builder;
pub mod factories;

/// LGDT, a less greedy decision tree learner.
///
/// Like CART, it builds the tree top-down one test at a time, but it chooses
/// each test by solving a depth-2 tree at the node with a
/// [`OptimalDepth2Tree`] solver and keeping its root. This two-level
/// lookahead is cheap thanks to the depth-2 solver and makes the tree much
/// less myopic than a purely greedy one.
///
/// Kiossou, Schaus, Nijssen and Aglin, *Efficient Lookahead Decision Trees*
/// (IDA 2024). Build one with [`LGDTBuilder`](builder::LGDTBuilder) or the
/// functions in [`factories`].
pub struct LGDT<D>
where
    D: OptimalDepth2Tree + ?Sized,
{
    search: Box<D>,
    config: BaseSearchConfig,
    tree: Tree,
}

impl<D> TreeSearchAlgorithm for LGDT<D>
where
    D: OptimalDepth2Tree + ?Sized,
{
    fn fit(&mut self, cover: &mut Cover) -> Result<(), FitError> {
        let depth = self.config.max_depth.min(2);
        let root_tree = match self.search.fit(self.config.min_support, depth, cover, None) {
            // No split beats a leaf (e.g. a single class): the tree is a leaf.
            Err(FitError::EmptyTree | FitError::EmptyCandidates) => {
                self.tree = self.leaf_tree(cover);
                return Ok(());
            }
            result => result?,
        };
        if self.config.max_depth <= 2 {
            self.tree = root_tree;
            return Ok(());
        }

        let mut solution_tree = Tree::new();
        let root_index = solution_tree.add_default_root();

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
        solution_tree.clean_orphaned_nodes();
        self.tree = solution_tree;
        Ok(())
    }

    fn tree(&self) -> &Tree {
        &self.tree
    }
}

impl<D> LGDT<D>
where
    D: OptimalDepth2Tree + ?Sized,
{
    /// Grows the subtree below `parent`, which tests `attribute`, with
    /// `depth` levels left. Returns the error of the subtree.
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
                    Err(FitError::EmptyTree) | Err(FitError::EmptyCandidates) => {
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
                        // A perfect depth-2 subtree is kept whole; otherwise
                        // only its root is kept and the recursion continues.
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

    /// A tree that is a single leaf over all of `cover`.
    fn leaf_tree(&self, cover: &mut Cover) -> Tree {
        let mut tree = Tree::new();
        let root = tree.add_default_root();
        tree.update_leaf_node(root, self.search.error(&cover.labels_count()));
        tree
    }

    /// The settings of the search.
    pub fn config(&self) -> &BaseSearchConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use crate::algorithms::greedy::lgdt::factories::with_error_minimizer;
    use crate::algorithms::TreeSearchAlgorithm;
    use crate::reader::data_reader::DataReader;
    use std::path::Path;

    #[test]
    fn test_d2_lgdt() {
        let reader = DataReader::default();
        let path = Path::new("test_data/anneal.txt");
        let cover_result = reader.read_file(path);

        let mut cover = cover_result.expect("the test data is readable");

        let mut lgdt = with_error_minimizer()
            .min_support(1)
            .max_depth(8)
            .build()
            .unwrap();
        lgdt.fit(&mut cover).unwrap();
        assert!(lgdt.tree.root_error() < cover.count() as f64);
    }
}
