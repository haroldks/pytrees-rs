use crate::globals::{float_is_null, get_tree_root_error, item};
use crate::searches::errors::{ErrorWrapper, NativeError};
use crate::searches::optimal::d2::GenericDepth2;
use crate::searches::utils::{Constraints, SearchStrategy};
use crate::searches::Statistics;
use crate::structures::Structure;
use crate::tree::{NodeInfos, Tree, TreeNode};

pub struct LGDT {
    pub error: f64,
    pub constraints: Constraints,
    pub statistics: Statistics,
    search_method: GenericDepth2,
    error_function: NativeError,
    pub tree: Tree,
}

impl LGDT {
    pub fn new(min_sup: usize, max_depth: usize, strategy: SearchStrategy) -> Self {
        let constraints = Constraints {
            max_depth,
            min_sup,
            search_strategy: strategy,
            ..Default::default()
        };

        Self {
            error: <f64>::INFINITY,
            constraints: Constraints {
                max_depth,
                min_sup,
                search_strategy: strategy,
                ..Default::default()
            },
            statistics: Statistics {
                constraints,
                ..Statistics::default()
            },
            search_method: GenericDepth2::new(strategy),
            error_function: NativeError::default(),
            tree: Tree::default(),
        }
    }

    pub fn fit<S>(&mut self, structure: &mut S)
    where
        S: Structure,
    {
        if self.constraints.max_depth <= 2 {
            let tree = self.search_method.fit(
                self.constraints.min_sup,
                self.constraints.max_depth,
                structure,
            );
            self.tree = tree;
        } else {
            let mut solution_tree = Tree::new();

            let root_tree = self
                .search_method
                .fit(self.constraints.min_sup, 2, structure);
            let mut root_attribute = None;

            if let Some(root) = root_tree.get_node(root_tree.get_root_index()) {
                solution_tree.add_root(TreeNode {
                    value: root.value,
                    index: 0,
                    left: 0,
                    right: 0,
                });
                root_attribute = root.value.test;
            }
            if root_attribute.is_some() {
                let root_index = solution_tree.get_root_index();
                self.recursion(
                    self.constraints.max_depth - 1,
                    structure,
                    &mut solution_tree,
                    root_index,
                    root_attribute,
                );
            }

            self.tree = solution_tree;
        }

        self.error = get_tree_root_error(&self.tree);
        self.update_statistics(structure)
    }

    fn recursion<S>(
        &mut self,
        depth: usize,
        structure: &mut S,
        tree: &mut Tree,
        index: usize,
        attribute: Option<usize>,
    ) -> f64
    where
        S: Structure,
    {
        return if depth <= 1 {
            let mut parent_error = 0.0;
            for (i, val) in [false, true].iter().enumerate() {
                let _ = structure.push(item(attribute.unwrap(), i));
                let child_tree = self
                    .search_method
                    .fit(self.constraints.min_sup, depth, structure);
                let child_error = get_tree_root_error(&child_tree);

                if child_error.is_infinite() {
                    let child_error = self.create_leaf(tree, structure, index, !*val);

                    parent_error += child_error;
                } else {
                    let child_index = self.create_child(tree, index, !*val);
                    self.move_tree(tree, child_index, &child_tree, child_tree.get_root_index());
                    parent_error += child_error;
                }

                structure.backtrack();
            }
            if let Some(parent) = tree.get_node_mut(index) {
                parent.value.error = parent_error;
            }
            parent_error
        } else {
            let mut parent_error = 0.0;
            for (i, val) in [false, true].iter().enumerate() {
                let _ = structure.push(item(attribute.unwrap(), i));
                let child_tree = self
                    .search_method
                    .fit(self.constraints.min_sup, 2, structure);
                // child_tree.print();
                let mut child_error = get_tree_root_error(&child_tree);
                if child_error.is_infinite() {
                    child_error = self.create_leaf(tree, structure, index, !*val);
                } else {
                    let child_index = self.create_child(tree, index, !*val);
                    if float_is_null(child_error) {
                        self.move_tree(tree, child_index, &child_tree, child_tree.get_root_index());
                    } else if let Some(child) = tree.get_node_mut(child_index) {
                        let mut child_next = None;
                        if let Some(root) = child_tree.get_node(child_tree.get_root_index()) {
                            child.value = root.value;
                            child_next = child.value.test;
                        }
                        child_error =
                            self.recursion(depth - 1, structure, tree, child_index, child_next);
                    }
                }
                parent_error += child_error;
                structure.backtrack();
            }
            if let Some(parent) = tree.get_node_mut(index) {
                parent.value.error = parent_error;
            }
            parent_error
        };
    }

    fn create_child(&self, tree: &mut Tree, parent: usize, is_left: bool) -> usize {
        let value = NodeInfos::default();
        let node = TreeNode::new(value);
        tree.add_node(parent, is_left, node)
    }

    fn create_leaf<S>(
        &self,
        tree: &mut Tree,
        structure: &mut S,
        parent: usize,
        is_left: bool,
    ) -> f64
    where
        S: Structure,
    {
        let leaf_index = self.create_child(tree, parent, is_left);
        let classes_support = structure.labels_support();
        let error = self.error_function.compute(classes_support);
        if let Some(leaf) = tree.get_node_mut(leaf_index) {
            leaf.value.error = error.0;
            leaf.value.out = Some(error.1)
        }
        error.0
    }

    fn move_tree(
        &self,
        dest_tree: &mut Tree,
        dest_index: usize,
        source_tree: &Tree,
        source_index: usize,
    ) {
        if let Some(source_node) = source_tree.get_node(source_index) {
            if let Some(root) = dest_tree.get_node_mut(dest_index) {
                root.value = source_node.value;
            }
            let source_left_index = source_node.left;

            if source_left_index > 0 {
                let mut left_index = 0;
                if let Some(root) = dest_tree.get_node_mut(dest_index) {
                    left_index = root.left;
                    if left_index == 0 {
                        left_index = self.create_child(dest_tree, dest_index, true);
                    }
                }
                self.move_tree(dest_tree, left_index, source_tree, source_left_index)
            }

            let source_right_index = source_node.right;
            if source_right_index > 0 {
                let mut right_index = 0;
                if let Some(root) = dest_tree.get_node_mut(dest_index) {
                    right_index = root.right;
                    if right_index == 0 {
                        right_index = self.create_child(dest_tree, dest_index, false);
                    }
                }
                self.move_tree(dest_tree, right_index, source_tree, source_right_index)
            }
        }
    }

    fn update_statistics<S: Structure>(&mut self, structure: &mut S) {
        self.statistics.tree_error = self.error;
        self.statistics.num_samples = structure.support();
        self.statistics.num_attributes = structure.num_attributes();
    }
}

#[cfg(test)]
mod test_lgdt {
    use crate::data::{BinaryData, FileReader};
    use crate::searches::errors::NativeError;
    use crate::searches::greedy::lgdt::LGDT;
    use crate::searches::optimal::d2::Murtree;
    use crate::searches::optimal::Depth2Algorithm;
    use crate::searches::utils::SearchStrategy;
    use crate::structures::Bitset;

    #[test]
    fn test_d2_lgdt() {
        let data = BinaryData::read("test_data/anneal.txt", false, 0.0);
        let mut structure = Bitset::new(&data);
        let error_function = Box::<NativeError>::default();

        let mut lgdt = LGDT::new(1, 5, SearchStrategy::LessGreedyMurtree);
        lgdt.fit(&mut structure);
        lgdt.tree.print()
    }
}
