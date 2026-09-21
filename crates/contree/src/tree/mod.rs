use serde::{Deserialize, Serialize};
use std::fmt;

/// Why a tree could not be walked or did not hold its invariant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeError {
    /// The tree has no nodes: `fit` has not run, or it failed.
    Empty,
    /// A node breaks the leaf/internal invariant. See [`Tree::validate`].
    Malformed { node: usize, reason: &'static str },
    /// A node tests a feature the instance does not have.
    FeatureOutOfRange {
        node: usize,
        feature: usize,
        n_features: usize,
    },
    /// A batch whose length is not a whole number of rows.
    RaggedInput { len: usize, n_features: usize },
}

impl fmt::Display for TreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TreeError::Empty => write!(f, "the tree is empty"),
            TreeError::Malformed { node, reason } => write!(f, "node {node}: {reason}"),
            TreeError::FeatureOutOfRange {
                node,
                feature,
                n_features,
            } => write!(
                f,
                "node {node} tests feature {feature}, but the instance has {n_features}"
            ),
            TreeError::RaggedInput { len, n_features } => write!(
                f,
                "{len} values is not a whole number of rows of {n_features} features"
            ),
        }
    }
}

impl std::error::Error for TreeError {}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct NodeInfos {
    // Specific data for decision trees
    pub feature: Option<usize>,
    pub error: usize,
    pub split: Option<f64>,
    pub label: Option<usize>,
}

impl Default for NodeInfos {
    fn default() -> Self {
        NodeInfos::new()
    }
}

impl NodeInfos {
    pub fn new() -> NodeInfos {
        NodeInfos {
            feature: None,
            error: usize::MAX,
            split: None,
            label: None,
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
            .unwrap_or_default()
    }

    pub fn node_details(&self, index: usize) -> NodeInfos {
        self.get_node(index)
            .map(|node| node.value)
            .unwrap_or_default()
    }

    pub fn root_error(&self) -> usize {
        self.get_node(self.get_root_index())
            .map(|node| node.value.error)
            .unwrap_or(usize::MAX) // Or another sensible default
    }

    pub fn node_error(&self, index: usize) -> usize {
        self.get_node(index)
            .map(|node| node.value.error)
            .unwrap_or(usize::MAX)
    }

    pub fn node_split(&self, index: usize) -> Option<f64> {
        self.get_node(index)
            .map(|node| node.value.split)
            .unwrap_or(None)
    }

    pub fn root_label(&self) -> Option<usize> {
        self.get_node(self.get_root_index())
            .and_then(|node| node.value.label)
    }

    pub fn node_label(&self, index: usize) -> Option<usize> {
        self.get_node(index).and_then(|node| node.value.label)
    }

    pub fn root_feature(&self) -> Option<usize> {
        self.get_node(self.get_root_index())
            .and_then(|node| node.value.feature)
    }

    pub fn root_split(&self) -> Option<f64> {
        self.get_node(self.get_root_index())
            .and_then(|node| node.value.split)
    }

    pub fn node_feature(&self, index: usize) -> Option<usize> {
        self.get_node(index).and_then(|node| node.value.feature)
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

    pub fn update_leaf_node(&mut self, index: usize, error: (usize, usize)) -> &mut Self {
        if let Some(updater) = self.update_node(index) {
            updater.error(error.0).label(error.1); // Maybe not the leaf
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
            } else if dest != 0 {
                // The source node is a leaf. The destination is usually a slot
                // in a pre-allocated skeleton, so it still points at children
                // that are now unreachable; leaving them attached is what made
                // a leaf indistinguishable from an internal node.
                self.detach_child(index, branch_value == 0);
            }
        }
    }

    fn detach_child(&mut self, index: usize, left: bool) {
        if let Some(node) = self.get_node_mut(index) {
            if left {
                node.left = 0;
            } else {
                node.right = 0;
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
            self.update_node(index).map(|updater| updater.leaf());
            return;
        }

        // Two leaves that predict the same label make their parent's test
        // pointless. The labels have to actually exist: comparing the two
        // `Option`s alone let `None == None` through, and the unwrap that
        // followed panicked on any node the search never filled in.
        if has_children && self.is_leaf(left) && self.is_leaf(right) {
            match (self.node_label(left), self.node_label(right)) {
                (Some(left_label), Some(right_label)) if left_label == right_label => {
                    self.update_node(index)
                        .map(|updater| updater.label(left_label).leaf());
                }
                _ => {}
            }
        }
    }

    fn is_leaf(&self, index: usize) -> bool {
        let (left, right) = self.node_children(index);
        (left == 0) && (right == 0)
    }

    fn can_be_leaf(&self, index: usize) -> bool {
        self.node_feature(index).is_none() && self.node_label(index).is_some()
    }

    /// The nodes of the arena, in insertion order. Index 0 is the root.
    ///
    /// Index 0 doubles as "no child", so a child index of 0 means the node has
    /// no child on that side, never "the root is my child".
    pub fn nodes(&self) -> &[TreeNode] {
        &self.tree
    }

    /// Rewrites the tree into its canonical form: a node is a leaf exactly
    /// when its `feature` is `None`, and a leaf carries no split and no
    /// children.
    ///
    /// The two searches spell a leaf differently -- the cache path writes
    /// `feature: Some(usize::MAX)` and `split: Some(f64::INFINITY)`, the
    /// depth-2 solver leaves a pre-allocated skeleton's children attached --
    /// so every tree passes through here before it reaches a caller.
    pub fn normalize_leaves(&mut self) {
        if self.is_empty() {
            return;
        }
        self.normalize_from(self.get_root_index());
    }

    fn normalize_from(&mut self, index: usize) {
        let (feature, left, right) = match self.get_node(index) {
            Some(node) => (node.value.feature, node.left, node.right),
            None => return,
        };

        let is_leaf = match feature {
            None => true,
            Some(usize::MAX) => true,
            Some(_) => left == 0 && right == 0,
        };

        if is_leaf {
            if let Some(node) = self.get_node_mut(index) {
                node.value.feature = None;
                node.value.split = None;
                node.left = 0;
                node.right = 0;
            }
            return;
        }

        if left != 0 {
            self.normalize_from(left);
        }
        if right != 0 {
            self.normalize_from(right);
        }
    }

    /// Checks the invariant `normalize_leaves` establishes.
    ///
    /// Every node is either a leaf -- no feature, no split, no children, and a
    /// label to predict -- or an internal node with a feature, a finite split
    /// threshold and two children.
    pub fn validate(&self) -> Result<(), TreeError> {
        if self.is_empty() {
            return Err(TreeError::Empty);
        }
        self.validate_from(self.get_root_index(), 0)
    }

    fn validate_from(&self, index: usize, depth: usize) -> Result<(), TreeError> {
        if depth > self.tree.len() {
            return Err(TreeError::Malformed {
                node: index,
                reason: "the tree contains a cycle",
            });
        }
        let node = self.get_node(index).ok_or(TreeError::Malformed {
            node: index,
            reason: "child index points outside the arena",
        })?;
        let (left, right) = (node.left, node.right);

        match node.value.feature {
            None => {
                if left != 0 || right != 0 {
                    return Err(TreeError::Malformed {
                        node: index,
                        reason: "a leaf must not have children",
                    });
                }
                if node.value.split.is_some() {
                    return Err(TreeError::Malformed {
                        node: index,
                        reason: "a leaf must not carry a split threshold",
                    });
                }
                if node.value.label.is_none() {
                    return Err(TreeError::Malformed {
                        node: index,
                        reason: "a leaf must carry a label",
                    });
                }
                Ok(())
            }
            Some(_) => {
                if left == 0 || right == 0 {
                    return Err(TreeError::Malformed {
                        node: index,
                        reason: "an internal node must have two children",
                    });
                }
                match node.value.split {
                    Some(split) if split.is_finite() => {}
                    _ => {
                        return Err(TreeError::Malformed {
                            node: index,
                            reason: "an internal node must carry a finite split threshold",
                        })
                    }
                }
                self.validate_from(left, depth + 1)?;
                self.validate_from(right, depth + 1)
            }
        }
    }

    /// Classifies one instance.
    ///
    /// **Routing convention: a split sends an instance left when
    /// `x[feature] <= threshold`, and right otherwise,** as in scikit-learn. This is the rule the
    /// search itself partitions by; getting it backwards silently produces a
    /// tree whose reported error has nothing to do with its predictions.
    pub fn predict_one(&self, x: &[f64]) -> Result<usize, TreeError> {
        Ok(self.tree[self.leaf_for(x)?]
            .value
            .label
            .expect("checked by leaf_for"))
    }

    /// The indices of the nodes an instance visits, root first, leaf last.
    pub fn decision_path(&self, x: &[f64]) -> Result<Vec<usize>, TreeError> {
        let mut path = Vec::new();
        let mut index = self.first_node()?;
        loop {
            path.push(index);
            match self.step(index, x)? {
                Some(next) => index = next,
                None => return Ok(path),
            }
        }
    }

    /// Classifies a batch. `rows` is row-major, `n_features` values per row.
    pub fn predict(&self, rows: &[f64], n_features: usize) -> Result<Vec<usize>, TreeError> {
        if n_features == 0 || rows.len() % n_features != 0 {
            return Err(TreeError::RaggedInput {
                len: rows.len(),
                n_features,
            });
        }
        rows.chunks_exact(n_features)
            .map(|x| self.predict_one(x))
            .collect()
    }

    fn first_node(&self) -> Result<usize, TreeError> {
        if self.is_empty() {
            return Err(TreeError::Empty);
        }
        Ok(self.get_root_index())
    }

    fn leaf_for(&self, x: &[f64]) -> Result<usize, TreeError> {
        let mut index = self.first_node()?;
        // The arena has one node per index, so a walk longer than that is a
        // cycle rather than a very deep tree.
        for _ in 0..=self.tree.len() {
            match self.step(index, x)? {
                Some(next) => index = next,
                None => return Ok(index),
            }
        }
        Err(TreeError::Malformed {
            node: index,
            reason: "the tree contains a cycle",
        })
    }

    /// One step of the walk: `None` means `index` is a leaf.
    fn step(&self, index: usize, x: &[f64]) -> Result<Option<usize>, TreeError> {
        let node = self.get_node(index).ok_or(TreeError::Malformed {
            node: index,
            reason: "child index points outside the arena",
        })?;

        let Some(feature) = node.value.feature else {
            if node.value.label.is_none() {
                return Err(TreeError::Malformed {
                    node: index,
                    reason: "a leaf must carry a label",
                });
            }
            return Ok(None);
        };

        if feature >= x.len() {
            return Err(TreeError::FeatureOutOfRange {
                node: index,
                feature,
                n_features: x.len(),
            });
        }
        let Some(split) = node.value.split else {
            return Err(TreeError::Malformed {
                node: index,
                reason: "an internal node must carry a split threshold",
            });
        };

        let next = if x[feature] <= split {
            node.left
        } else {
            node.right
        };
        if next == 0 {
            return Err(TreeError::Malformed {
                node: index,
                reason: "an internal node must have two children",
            });
        }
        Ok(Some(next))
    }

    pub fn print(&self) {
        let mut stack: Vec<(usize, Option<&TreeNode>)> = Vec::new();
        let root = self.get_node(self.get_root_index());
        stack.push((0, root));
        while let Some((deep, node_opt)) = stack.pop() {
            if let Some(node) = node_opt {
                for _ in 0..deep {
                    print!("    ");
                }
                println!("----{:?}", node.value);

                stack.push((deep + 1, self.get_right_child(node)));
                stack.push((deep + 1, self.get_left_child(node)));
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

    pub fn error(self, error: usize) -> Self {
        self.node.value.error = error;
        self
    }

    pub fn split(self, metric: f64) -> Self {
        self.node.value.split = Some(metric);
        self
    }

    pub fn label(self, output: usize) -> Self {
        self.node.value.label = Some(output);
        self
    }

    pub fn feature(self, test: usize) -> Self {
        self.node.value.feature = Some(test);
        self
    }

    /// Turns the node into a leaf: no test, no split, no children.
    ///
    /// `feature: None` is *the* definition of a leaf everywhere in the crate,
    /// so clearing it here is not optional -- it used to be a commented-out
    /// line, which is how three different leaf spellings came to exist.
    pub fn leaf(self) -> Self {
        self.node.value.feature = None;
        self.node.value.split = None;
        self.node.left = 0;
        self.node.right = 0;
        self
    }

    pub fn get_children(&self) -> (usize, usize) {
        (self.node.left, self.node.right)
    }
}
