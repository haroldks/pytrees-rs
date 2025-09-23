use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct NodeInfos {
    // Specific data for decision trees
    pub test: Option<usize>,
    pub error: f64,
    pub metric: Option<f64>,
    pub out: Option<f64>,
}

impl Default for NodeInfos {
    fn default() -> Self {
        NodeInfos::new()
    }
}

impl NodeInfos {
    pub fn new() -> NodeInfos {
        NodeInfos {
            test: None,
            error: <f64>::INFINITY,
            metric: None,
            out: None,
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize, Debug, Default)]
pub struct TreeNode {
    pub value: NodeInfos,
    pub index: usize,
    pub left: usize,
    pub right: usize,
}

impl TreeNode {
    pub fn new(value: NodeInfos) -> TreeNode {
        TreeNode {
            value,
            index: 0,
            left: 0,
            right: 0,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Tree {
    tree: Vec<TreeNode>,
}

impl Default for Tree {
    fn default() -> Self {
        Self::new()
    }
}

impl Tree {
    pub fn new() -> Self {
        Tree { tree: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Tree {
            tree: Vec::with_capacity(capacity),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    pub fn len(&self) -> usize {
        self.tree.len()
    }

    pub fn actual_len(&self) -> usize {
        self.count_node_recursion(self.get_root_index())
    }

    // ! Is it still relevant
    fn count_node_recursion(&self, node_index: usize) -> usize {
        let mut left_index = 0;
        let mut right_index = 0;
        if let Some(node) = self.get_node(node_index) {
            if node.left == node.right {
                return 1;
            } else {
                left_index = node.left;
                right_index = node.right;
            }
        }

        let mut count = 0;

        if left_index != 0 {
            count += self.count_node_recursion(left_index);
        }
        if right_index != 0 {
            count += self.count_node_recursion(right_index);
        }

        count + 1
    }

    pub fn add_node(&mut self, parent: usize, is_left: bool, mut node: TreeNode) -> usize {
        node.index = self.tree.len();
        self.tree.push(node);
        let position = self.tree.len() - 1;
        if position == 0 {
            return position;
        }
        if let Some(parent_node) = self.tree.get_mut(parent) {
            if is_left {
                parent_node.left = position
            } else {
                parent_node.right = position
            }
        };
        position
    }

    pub fn add_root(&mut self, root: TreeNode) -> usize {
        self.add_node(0, false, root)
    }

    pub fn add_default_root(&mut self) -> usize {
        self.add_root(TreeNode::default())
    }

    pub fn add_left_node(&mut self, parent: usize, node: TreeNode) -> usize {
        self.add_node(parent, true, node)
    }
    pub fn add_right_node(&mut self, parent: usize, node: TreeNode) -> usize {
        self.add_node(parent, false, node)
    }

    pub fn create_child(&mut self, parent: usize, left: bool) -> usize {
        self.add_node(parent, left, TreeNode::default())
    }

    pub fn get_root_index(&self) -> usize {
        0
    }

    pub fn get_node(&self, index: usize) -> Option<&TreeNode> {
        self.tree.get(index)
    }

    pub fn get_node_mut(&mut self, index: usize) -> Option<&mut TreeNode> {
        self.tree.get_mut(index)
    }

    pub fn get_left_child(&self, node: &TreeNode) -> Option<&TreeNode> {
        if node.left == 0 {
            None
        } else {
            self.tree.get(node.left)
        }
    }
    pub fn get_left_child_mut(&mut self, node: &TreeNode) -> Option<&mut TreeNode> {
        if node.left == 0 {
            None
        } else {
            self.tree.get_mut(node.left)
        }
    }

    pub fn get_right_child(&self, node: &TreeNode) -> Option<&TreeNode> {
        if node.right == 0 {
            None
        } else {
            self.tree.get(node.right)
        }
    }
    pub fn get_right_child_mut(&mut self, node: &TreeNode) -> Option<&mut TreeNode> {
        if node.right == 0 {
            None
        } else {
            self.tree.get_mut(node.right)
        }
    }

    pub fn empty_tree(depth: usize) -> Tree {
        let mut tree = Tree::new();
        let value = NodeInfos::new();
        let node = TreeNode::new(value);
        let root = tree.add_root(node);
        Self::build_tree_recurse(&mut tree, root, depth);
        tree
    }

    fn build_tree_recurse(tree: &mut Tree, parent: usize, depth: usize) {
        if depth == 0 {
            if let Some(parent_node) = tree.get_node_mut(parent) {
                parent_node.left = 0;
                parent_node.right = 0;
            }
        } else {
            let value = NodeInfos::new();
            let node = TreeNode::new(value);
            let left = tree.add_node(parent, true, node);
            Self::build_tree_recurse(tree, left, depth - 1);
            let node = TreeNode::new(value);
            let right = tree.add_node(parent, false, node);
            Self::build_tree_recurse(tree, right, depth - 1);
        }
    }

    // New functions

    pub fn root_details(&self) -> NodeInfos {
        self.get_node(self.get_root_index())
            .map(|node| node.value)
            .unwrap_or(NodeInfos::default())
    }

    pub fn node_details(&self, index: usize) -> NodeInfos {
        self.get_node(index)
            .map(|node| node.value)
            .unwrap_or(NodeInfos::default())
    }

    pub fn root_error(&self) -> f64 {
        self.get_node(self.get_root_index())
            .map(|node| node.value.error)
            .unwrap_or(f64::MAX) // Or another sensible default
    }

    pub fn node_error(&self, index: usize) -> f64 {
        self.get_node(index)
            .map(|node| node.value.error)
            .unwrap_or(f64::MAX)
    }

    pub fn node_metric(&self, index: usize) -> Option<f64> {
        self.get_node(index)
            .map(|node| node.value.metric)
            .unwrap_or(Some(0.0))
    }

    pub fn root_output(&self) -> Option<f64> {
        self.get_node(self.get_root_index())
            .and_then(|node| node.value.out)
    }

    pub fn node_output(&self, index: usize) -> Option<f64> {
        self.get_node(index).and_then(|node| node.value.out)
    }

    pub fn root_test(&self) -> Option<usize> {
        self.get_node(self.get_root_index())
            .and_then(|node| node.value.test)
    }

    pub fn node_test(&self, index: usize) -> Option<usize> {
        self.get_node(index).and_then(|node| node.value.test)
    }

    pub fn update_node(&mut self, index: usize) -> Option<NodeUpdater> {
        self.get_node_mut(index).map(NodeUpdater::new)
    }

    pub fn update_root(&mut self) -> Option<NodeUpdater> {
        self.get_node_mut(0).map(NodeUpdater::new)
    }

    pub fn node_children(&self, index: usize) -> (usize, usize) {
        self.get_node(index)
            .map_or((0, 0), |node| (node.left, node.right))
    }

    pub fn update_leaf_node(&mut self, index: usize, error: (f64, f64)) -> &mut Self {
        if let Some(updater) = self.update_node(index) {
            updater.error(error.0).output(error.1); // Maybe not the leaf
        }
        self
    }

    pub fn update_subtree(&mut self, index: usize, origin: &Tree, origin_index: usize) {
        let (left_index, right_index) = self.update_node(index).map_or((0, 0), |updater| {
            updater
                .value(origin.node_details(origin_index))
                .get_children()
        });

        let (origin_left_index, origin_right_index) = origin.node_children(origin_index);

        for (branch_value, (&source, mut dest)) in [origin_left_index, origin_right_index]
            .iter()
            .zip([left_index, right_index])
            .enumerate()
        {
            if source > 0 {
                if dest == 0 {
                    dest = self.create_child(index, branch_value == 0);
                }
                self.update_subtree(dest, origin, source);
            }
        }
    }

    pub fn clean_orphaned_nodes(&mut self) {
        if self.is_empty() {
            return;
        }
        self.cleanup_leaves(self.get_root_index())

    }

    fn cleanup_leaves(&mut self, index: usize) {
        let (left, right) = self.node_children(index);
        if left != 0 {
            self.cleanup_leaves(left);
        }
        if right != 0 {
            self.cleanup_leaves(right)
        }

        let has_children = left != 0 || right != 0;
        if has_children && self.can_be_leaf(index) {
            self.update_node(index)
                .map(|updater| updater.leaf());
            return;
        }

        if has_children && self.is_leaf(left) && self.is_leaf(right) && self.node_output(left).eq(&self.node_output(right)) {
            let output = self.node_output(right).unwrap();
            self.update_node(index)
                .map(|updater| updater.output(output)
                    .clean_test()
                    .leaf());
        }
    }



    fn is_leaf(&self, index: usize) -> bool {
        let (left, right) = self.node_children(index);
        (left == 0) && (right == 0)
    }

    fn can_be_leaf(&self, index: usize) -> bool {
        self.node_test(index).is_none() && self.node_output(index).is_some()
    }



    pub fn print(&self) {
        let mut stack: Vec<(usize, Option<&TreeNode>)> = Vec::new();
        let root = self.get_node(self.get_root_index());
        stack.push((0, root));
        while !stack.is_empty() {
            let next = stack.pop();
            if let Some((deep, node_opt)) = next {
                if let Some(node) = node_opt {
                    for _i in 0..deep {
                        print!("    ");
                    }
                    println!("----{:?}", node.value);

                    stack.push((deep + 1, self.get_right_child(node)));
                    stack.push((deep + 1, self.get_left_child(node)));
                }
            }
        }
    }
}

pub struct NodeUpdater<'a> {
    node: &'a mut TreeNode,
}

impl<'a> NodeUpdater<'a> {
    pub fn new(node: &'a mut TreeNode) -> Self {
        Self { node }
    }

    pub fn value(self, value: NodeInfos) -> Self {
        self.node.value = value;
        self
    }

    pub fn error(self, error: f64) -> Self {
        self.node.value.error = error;
        self
    }

    pub fn metric(self, metric: f64) -> Self {
        self.node.value.metric = Some(metric);
        self
    }

    pub fn output(self, output: f64) -> Self {
        self.node.value.out = Some(output);
        self
    }

    pub fn test(self, test: usize) -> Self {
        self.node.value.test = Some(test);
        self
    }

    pub fn left_child(self, index: usize) -> Self {
        self.node.left = index;
        self
    }

    pub fn right_child(self, index: usize) -> Self {
        self.node.right = index;
        self
    }

    pub fn leaf(self) -> Self {
        // self.node.value.test = None;
        self.node.left = 0;
        self.node.right = 0;
        self
    }

    pub fn clean_test(self) -> Self {
        self.node.value.test = None;
        self
    }

    pub fn get_children(&self) -> (usize, usize) {
        (self.node.left, self.node.right)
    }


}

#[cfg(test)]
mod binary_tree_test {
    use crate::tree::{NodeInfos, Tree, TreeNode};

    #[test]
    fn create_node_data() {
        let data = NodeInfos::new();
        assert_eq!(data.error, <f64>::INFINITY);
        assert_eq!(data.test.is_none(), true);
        assert_eq!(data.out, None);
    }

    #[test]
    fn create_tree_node() {
        let data = NodeInfos::new();
        let node = TreeNode::new(data);
        assert_eq!(node.right, 0);
        assert_eq!(node.left, 0);
        assert_eq!(node.index, 0);
    }

    #[test]
    fn tree_default() {
        let tree = Tree::default();
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn tree_new() {
        let tree = Tree::default();
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn tree_is_empty() {
        let mut tree = Tree::new();
        assert_eq!(tree.is_empty(), true);

        let root = TreeNode::new(NodeInfos::default());
        tree.add_root(root);
        assert_eq!(tree.is_empty(), false);
    }

    #[test]
    fn binarytree_add_root() {
        let mut tree: Tree = Tree::new();
        let root = TreeNode::new(NodeInfos::default());
        let root_index = tree.add_root(root);
        assert_eq!(0, root_index);
    }

    #[test]
    fn binarytree_get_root_index() {
        let mut tree = Tree::new();
        let root = TreeNode::new(NodeInfos::default());
        let _ = tree.add_root(root);
        let root_index = tree.get_root_index();
        assert_eq!(0, root_index);
    }

    #[test]
    fn binarytree_get_left_child() {
        let mut tree = Tree::new();
        let root = TreeNode::new(NodeInfos::default());
        let root_index = tree.add_root(root);
        let node_infos = NodeInfos {
            test: Some(15),
            error: 0.0,
            metric: None,
            out: None,
        };
        let left_node = TreeNode::new(node_infos);
        let _ = tree.add_left_node(root_index, left_node);
        let root = tree.get_node(root_index).unwrap();
        let left_node = tree.get_left_child(root).unwrap();
        assert_eq!(left_node.value.test, Some(15));
    }

    #[test]
    fn binarytree_get_right_child() {
        let mut tree = Tree::new();
        let root = TreeNode::new(NodeInfos::default());
        let root_index = tree.add_root(root);
        let node_infos = NodeInfos {
            test: Some(15),
            error: 0.0,
            metric: None,
            out: None,
        };
        let right_node = TreeNode::new(node_infos);
        let _ = tree.add_right_node(root_index, right_node);
        let root = tree.get_node(root_index).unwrap();
        let right_node = tree.get_right_child(root).unwrap();
        assert_eq!(right_node.value.test, Some(15));
    }

    #[test]
    fn test_get_node() {
        let mut tree = Tree::new();
        let node_infos = NodeInfos {
            test: Some(33),
            error: 0.0,
            metric: None,
            out: None,
        };
        let root = TreeNode::new(node_infos);
        let _ = tree.add_root(root);
        let root_index = tree.get_root_index();
        let root = tree.get_node(root_index).unwrap();
        assert_eq!(Some(33), root.value.test)
    }

    #[test]
    fn test_get_node_mut() {
        let mut tree = Tree::new();
        let node_infos = NodeInfos {
            test: Some(55),
            error: 0.0,
            metric: None,
            out: None,
        };
        let root = TreeNode::new(node_infos);
        let _ = tree.add_root(root);
        let root_index = tree.get_root_index();
        let root = tree.get_node_mut(root_index).unwrap();
        root.value.test = Some(12);
        assert_eq!(Some(12), root.value.test);
    }

    #[test]
    fn test_add_left_node() {
        let mut tree = Tree::new();
        let node_infos = NodeInfos {
            test: Some(15),
            error: 0.0,
            metric: None,
            out: None,
        };
        let root = TreeNode::new(node_infos);
        let root_index = tree.add_root(root);
        let node_infos = NodeInfos {
            test: Some(11),
            error: 0.0,
            metric: None,
            out: None,
        };
        let left_node = TreeNode::new(node_infos);
        let _ = tree.add_left_node(root_index, left_node);
        let root = tree.get_node(root_index).unwrap();
        let left_node = tree.get_left_child(root).unwrap();
        assert_eq!(left_node.value.test, Some(11));
    }

    #[test]
    fn test_add_right_node() {
        let mut tree = Tree::new();
        let node_infos = NodeInfos {
            test: Some(15),
            error: 0.0,
            metric: None,
            out: None,
        };
        let root = TreeNode::new(node_infos);
        let root_index = tree.add_root(root);
        let node_infos = NodeInfos {
            test: Some(22),
            error: 0.0,
            metric: None,
            out: None,
        };
        let right_node = TreeNode::new(node_infos);
        let _ = tree.add_right_node(root_index, right_node);
        let root = tree.get_node(root_index).unwrap();
        let right_node = tree.get_right_child(root).unwrap();
        assert_eq!(right_node.value.test, Some(22));
    }
}
