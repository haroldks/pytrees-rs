mod conditions;
mod similarity;
mod config;
mod builder;

use crate::cache::{CacheEntry, Caching};
use crate::globals::{attribute, float_is_null, get_tree_root_error, item};
use crate::heuristics::Heuristic;
use crate::searches::errors::ErrorWrapper;
use crate::searches::optimal::d2::Murtree;
use crate::searches::optimal::dl85::conditions::StopConditions;
use crate::searches::optimal::dl85::similarity::SimilarityCover;
use crate::searches::optimal::Depth2Algorithm;
use crate::searches::utils::{
    BranchingStrategy, CacheInitStrategy, Constraints, LowerBoundStrategy, NodeExposedData,
    SearchStrategy, Specialization, Statistics, StopReason,
};
use crate::structures::Structure;
use crate::tree::NodeInfos;
use crate::tree::{Tree, TreeNode};
use std::collections::BTreeSet;
use std::time::Instant;

// The search will return the node error, the reason the search was stop and if we did a projection in the database
pub type SearchReturn = (f64, StopReason, bool);
pub type BranchChoice = (usize, f64, f64);
pub struct DL85<C, E, H>
where
    C: Caching + ?Sized,
    E: ErrorWrapper + ?Sized,
    H: Heuristic + ?Sized,
{
    constraints: Constraints,
    pub statistics: Statistics,
    stop_conditions: StopConditions,
    cache: Box<C>,
    error_function: Box<E>,
    heuristic: Box<H>,
    pub tree: Tree,
    runtime: Instant,
    murtree: Murtree // WARNING Only work for classification,
}

impl<C, E, H> DL85<C, E, H>
where
    C: Caching + ?Sized,
    E: ErrorWrapper + ?Sized,
    H: Heuristic + ?Sized,
{
    pub fn new(
        min_sup: usize,
        max_depth: usize,
        max_error: f64,
        max_time: usize,
        one_time_sort: bool,
        cache_init_size: usize,
        cache_init_strategy: CacheInitStrategy,
        specialization: Specialization,
        lower_bound_strategy: LowerBoundStrategy,
        branching: BranchingStrategy,
        data_format: NodeExposedData,
        cache: Box<C>,
        error_function: Box<E>,
        heuristic: Box<H>,
    ) -> Self {
        let constraints = Constraints {
            max_depth,
            min_sup,
            max_error,
            max_time,
            one_time_sort,
            specialization,
            node_exposed_data: data_format,
            lower_bound_strategy,
            branching_strategy: branching,
            cache_init_size,
            cache_init_strategy,
            discrepancy_budget: 0,
            search_strategy: SearchStrategy::None_,
        };

        Self {
            constraints,
            statistics: Statistics {
                constraints,
                ..Statistics::default()
            },
            stop_conditions: StopConditions::default(),
            cache,
            error_function,
            heuristic,
            tree: Tree::default(),
            runtime: Instant::now(),
            murtree: Murtree::default(),
        }
    }

    pub fn fit<S: Structure>(&mut self, structure: &mut S) {
        self.statistics.num_attributes = structure.num_attributes();
        self.statistics.num_samples = structure.support();

        // Init cache
        // TODO: This should take in strategy and init_capacity
        let root_index = self.cache.init();

        let root_error = self.error_as_leaf(structure);
        if let Some(root) = self.cache.set_root_infos() {
            //root.error = root_error.0;
            root.leaf_error = root_error.0;
            root.target = root_error.1;
            root.size = self.statistics.num_samples;
        }
        let bound = <f64>::min(root_error.0, self.constraints.max_error);

        // Collect the potential candidates based on the support constraint and sort them based on the heuristic
        let mut candidates = Vec::new();
        if self.constraints.min_sup == 1 {
            candidates = (0..structure.num_attributes()).collect();
        } else {
            for i in 0..structure.num_attributes() {
                if structure.temp_push(item(i, 0)) >= self.constraints.min_sup
                    && structure.temp_push(item(i, 1)) >= self.constraints.min_sup
                {
                    candidates.push(i);
                }
            }
        }

        self.heuristic.compute(structure, &mut candidates);

        let mut itemset = BTreeSet::new();

        let mut similarity = SimilarityCover::default();

        // Starting the search
        self.runtime = Instant::now();
        self.recursion(
            structure,
            0,
            bound,
            <usize>::MAX,
            &mut itemset,
            &candidates,
            root_index,
            true,
            &mut similarity,
        );

        self.update_statistics();
        self.get_solution_tree();
    }

    fn recursion<S: Structure>(
        &mut self,
        structure: &mut S,
        depth: usize,
        upper_bound: f64,
        parent_item: usize,
        itemset: &mut BTreeSet<usize>,
        candidates: &[usize],
        parent_index: Option<usize>,
        parent_is_new: bool,
        similarity: &mut SimilarityCover,
    ) -> SearchReturn {
        let mut child_upper_bound = upper_bound;
        let current_support = structure.support();
        self.statistics.search_space_size += 1;

        // BEGIN STEP: Check if we should stop

        if let Some(node) = self.cache.get(itemset, parent_index) {
            let return_condition = self.stop_conditions.check(
                node,
                current_support,
                self.constraints.min_sup,
                depth,
                self.constraints.max_depth,
                self.runtime.elapsed(),
                self.constraints.max_time,
                child_upper_bound,
            );

            if return_condition.0 {
                node.upper_bound = upper_bound;
                return (node.error, return_condition.1, false);
            }
        }

        if !parent_is_new {
            structure.push(parent_item);
        }

        // TODO: Implement the similarity
        if let LowerBoundStrategy::Similarity = self.constraints.lower_bound_strategy {
            if let Some(node) = self.cache.get(itemset, parent_index) {
                node.lower_bound =
                    <f64>::max(node.lower_bound, similarity.compute_similarity(structure));

                let return_condition = self
                    .stop_conditions
                    .stop_from_lower_bound(node, child_upper_bound);
                if return_condition.0 {
                    return (node.error, return_condition.1, true);
                }
            }
        }

        if self.constraints.max_depth - depth <= 2 {
            if let Specialization::Murtree = self.constraints.specialization {
                return self.apply_murtree_d2_odt(
                    structure,
                    parent_index,
                    upper_bound,
                    itemset,
                    self.constraints.max_depth - depth,
                );
            }
        }

        // BEGIN STEP: Get the node candidates
        let mut node_candidates =
            self.get_node_candidates(structure, attribute(parent_item), candidates);

        if node_candidates.is_empty() {
            if let Some(node) = self.cache.get(itemset, parent_index) {
                node.to_leaf();
                return (node.error, StopReason::None, true);
            }
        }

        if !self.constraints.one_time_sort {
            self.heuristic.compute(structure, &mut node_candidates);
        }

        let mut child_similarity_data = SimilarityCover::default();
        let mut min_lower_bound = <f64>::INFINITY;

        let mut parent_error = self
            .cache
            .get(itemset, parent_index)
            .map_or(<f64>::INFINITY, |infos| infos.error);

        for child in node_candidates.iter() {
            let branching_choice =
                self.branching_strategy(*child, itemset, structure, &mut child_similarity_data);

            let it = item(*child, branching_choice.0);
            itemset.insert(it);

            let (is_new, child_index) = self.cache.insert(itemset);

            // TODO : Move this in a function
            if is_new {
                let size = structure.push(it);
                let error = self.error_as_leaf(structure);
                if let Some(node) = self.cache.get(itemset, child_index) {
                    node.leaf_error = error.0;
                    node.target = error.1;
                    node.size = size;
                }
            } else {
                self.statistics.cache_callbacks += 1;
            }

            if let Some(node) = self.cache.get(itemset, child_index) {
                node.lower_bound = branching_choice.1;
            }

            let first_child_return = self.recursion(
                structure,
                depth + 1,
                child_upper_bound,
                it,
                itemset,
                &node_candidates,
                child_index,
                is_new,
                &mut child_similarity_data,
            );

            let left_error = first_child_return.0;

            // Now that the search is done. We have to see if the we new to go back to previous
            self.backtrack(
                structure,
                itemset,
                is_new,
                &it,
                &first_child_return,
                child_index,
                &mut child_similarity_data,
            );

            if left_error >= child_upper_bound - branching_choice.2 {
                if let Some(node) = self.cache.get(itemset, child_index) {
                    min_lower_bound = <f64>::min(
                        min_lower_bound,
                        match left_error.is_finite() {
                            true => left_error + branching_choice.2,
                            false => node.lower_bound + branching_choice.2,
                        },
                    );
                }
                itemset.remove(&it);
                continue;
            }

            // TODO : Watch out
            itemset.remove(&it);

            // Going to the left
            let right_upper_bound = child_upper_bound - left_error;
            let it = item(*child, (branching_choice.0 + 1) % 2);
            itemset.insert(it);

            let (is_new, child_index) = self.cache.insert(itemset);

            if is_new {
                let size = structure.push(it);
                let error = self.error_as_leaf(structure);
                if let Some(node) = self.cache.get(itemset, child_index) {
                    node.leaf_error = error.0;
                    node.target = error.1;
                    node.size = size;
                }
            } else {
                self.statistics.cache_callbacks += 1;
            }

            if let Some(node) = self.cache.get(itemset, child_index) {
                node.lower_bound = branching_choice.2;
            }

            let second_child_return = self.recursion(
                structure,
                depth + 1,
                right_upper_bound,
                it,
                itemset,
                &node_candidates,
                child_index,
                is_new,
                &mut child_similarity_data,
            );

            let right_error = second_child_return.0;

            // Now that the search is done. We have to see if the we new to go back to previous
            self.backtrack(
                structure,
                itemset,
                is_new,
                &it,
                &second_child_return,
                child_index,
                &mut child_similarity_data,
            );
            itemset.remove(&it);

            let feature_error = left_error + right_error;

            if feature_error < child_upper_bound {
                child_upper_bound = feature_error;
                parent_error = feature_error;

                if let Some(parent_node) = self.cache.get(itemset, parent_index) {
                    parent_node.error = child_upper_bound;
                    parent_node.test = *child;

                    if float_is_null(parent_node.lower_bound - child_upper_bound) {
                        parent_node.is_optimal = true;
                        parent_node.upper_bound = upper_bound;
                        return (parent_error, StopReason::Done, true);
                    }
                }
            } else {
                min_lower_bound = <f64>::min(feature_error, min_lower_bound);
            }
        }

        let mut node_error = 0.0;
        if let Some(node) = self.cache.get(itemset, parent_index) {
            node_error = node.error;
            node.is_optimal = true;
            node.upper_bound = upper_bound;
            if node.error.is_infinite() {
                node.lower_bound =
                    <f64>::max(node.lower_bound, <f64>::max(min_lower_bound, upper_bound));
            }
            return (node.error, StopReason::LowerBoundConstrained, true);
        }

        (node_error, StopReason::Done, true)
    }

    fn get_node_candidates<S: Structure>(
        &self,
        structure: &mut S,
        last_candidate: usize,
        candidates: &[usize],
    ) -> Vec<usize> {
        let mut node_candidates = Vec::new();
        let support = structure.support();
        for potential_candidate in candidates {
            if *potential_candidate == last_candidate {
                continue;
            }
            let left_support = structure.temp_push(item(*potential_candidate, 0));
            let right_support = support - left_support;

            if left_support >= self.constraints.min_sup && right_support >= self.constraints.min_sup
            {
                node_candidates.push(*potential_candidate);
            }
        }
        node_candidates
    }

    fn get_children_stored_lower_bounds(
        &self,
        attribute: usize,
        itemset: &mut BTreeSet<usize>,
    ) -> [f64; 2] {
        let mut lower_bounds = [0.0; 2];
        for (i, lower_bound) in lower_bounds.iter_mut().enumerate() {
            itemset.insert(item(attribute, i));
            if let Some(node) = self.cache.find(itemset) {
                let error = node.error;
                *lower_bound = match error.is_finite() {
                    true => error,
                    false => node.lower_bound,
                };
            }
            itemset.remove(&item(attribute, i));
        }
        lower_bounds
    }

    fn error_as_leaf<S: Structure>(&self, structure: &mut S) -> (f64, f64) {
        let error = match self.constraints.node_exposed_data {
            NodeExposedData::ClassesSupport => {
                self.error_function.compute(structure.labels_support())
            }
            NodeExposedData::Tids => self.error_function.compute(&structure.get_tids()),
        };
        error
    }

    fn comput_similarity_lower_bounds<S: Structure>(
        &self,
        lower_bounds: &mut [f64; 2],
        attribute: usize,
        similarity_cover: &mut SimilarityCover,
        structure: &mut S,
    ) {
        for (branch, lower_bound) in lower_bounds.iter_mut().enumerate() {
            structure.push(item(attribute, branch));
            let similarity_lower_bound = similarity_cover.compute_similarity(structure);
            *lower_bound = <f64>::max(*lower_bound, similarity_lower_bound);
            structure.backtrack();
        }
    }

    fn branching_strategy<S: Structure>(
        &self,
        child: usize,
        itemset: &mut BTreeSet<usize>,
        structure: &mut S,
        similarity_dataset: &mut SimilarityCover,
    ) -> BranchChoice {
        let mut lower_bounds = [0.0, 0.0];
        // If Dynamic branching is enabled, we check where to move first
        if let BranchingStrategy::Dynamic = self.constraints.branching_strategy {
            lower_bounds = self.get_children_stored_lower_bounds(child, itemset);

            if let LowerBoundStrategy::Similarity = self.constraints.lower_bound_strategy {
                self.comput_similarity_lower_bounds(
                    &mut lower_bounds,
                    child,
                    similarity_dataset,
                    structure,
                );
            }
        }

        let first_item_type = (lower_bounds[0] > lower_bounds[1]) as usize;
        let first_lower_bound = lower_bounds[first_item_type];
        let second_lower_bound = lower_bounds[(first_item_type + 1) % 2];
        (first_item_type, first_lower_bound, second_lower_bound)
    }

    fn backtrack<S: Structure>(
        &mut self,
        structure: &mut S,
        itemset: &mut BTreeSet<usize>,
        is_new: bool,
        item: &usize,
        return_infos: &SearchReturn,
        child_index: Option<usize>,
        child_similarity_data: &mut SimilarityCover,
    ) {
        let has_intersected = return_infos.2;

        if !(is_new || has_intersected) {
            structure.push(*item);
        }
        if let LowerBoundStrategy::Similarity = self.constraints.lower_bound_strategy {
            if !matches!(return_infos.1, StopReason::LowerBoundConstrained) {
                if let Some(node) = self.cache.get(itemset, child_index) {
                    child_similarity_data.update(node.error, structure);
                }
            }
        }
        structure.backtrack();
    }
    fn update_statistics(&mut self) {
        self.statistics.cache_size = self.cache.size();
        self.statistics.duration = self.runtime.elapsed();
        if let Some(infos) = self.cache.get_root_infos() {
            self.statistics.tree_error = infos.error;
        }
    }
    fn apply_murtree_d2_odt<S: Structure>(
        &mut self,
        structure: &mut S,
        index: Option<usize>,
        upper_bound: f64,
        itemset: &mut BTreeSet<usize>,
        depth: usize,
    ) -> SearchReturn {
        if let Some(node) = self.cache.get(itemset, index) {
            if upper_bound < node.lower_bound {
                return (node.error, StopReason::LowerBoundConstrained, true);
            }
        }
        let tree = self.murtree.fit(self.constraints.min_sup, depth, structure);
        let tree_error = get_tree_root_error(&tree);
        self.cache_murtree_results(itemset, index, &tree, tree.get_root_index());
        (tree_error, StopReason::FromSpecializedAlgorithm, true)
    }

    fn cache_murtree_results(
        &mut self,
        itemset: &mut BTreeSet<usize>,
        index: Option<usize>,
        tree: &Tree,
        tree_index: usize,
    ) {
        if let Some(tree_node) = tree.get_node(tree_index) {
            if let Some(cache_node) = self.cache.get(itemset, index) {
                cache_node.error = tree_node.value.error;
                cache_node.leaf_error = tree_node.value.error;

                if tree_node.value.test.is_none() {
                    cache_node.is_leaf = true;
                    cache_node.target = tree_node.value.out.unwrap_or(0.0);
                    return;
                } else {
                    cache_node.test = tree_node.value.test.unwrap_or(<usize>::MAX);
                }
            }
            for (branch, idx) in [tree_node.left, tree_node.right].iter().enumerate() {
                if *idx > 0 {
                    let it = item(tree_node.value.test.unwrap_or(<usize>::MAX), branch);
                    itemset.insert(it);
                    let (_, cache_child_index) = self.cache.insert(itemset);
                    self.cache_murtree_results(itemset, cache_child_index, tree, *idx);
                    itemset.remove(&it);
                }
            }
        }
    }

    fn create_solution_tree_entry(&self, cache_entry: &CacheEntry) -> NodeInfos {
        let mut infos = NodeInfos {
            error: cache_entry.error,
            ..Default::default()
        };
        match cache_entry.is_leaf {
            true => {
                infos.out = Some(cache_entry.target);
            }
            false => infos.test = Some(cache_entry.test),
        };
        infos
    }

    fn get_solution_tree(&mut self) {
        let mut tree = Tree::new();
        let mut path = BTreeSet::new();
        if let Some(cache_root) = self.cache.get_root_infos() {
            let infos = self.create_solution_tree_entry(cache_root);
            let root = tree.add_root(TreeNode::new(infos));
            self.get_solution_tree_recursion(cache_root.test, &mut path, &mut tree, root);
        }
        self.tree = tree;
    }
    fn get_solution_tree_recursion(
        &self,
        attribute: usize,
        path: &mut BTreeSet<usize>,
        tree: &mut Tree,
        index: usize,
    ) {
        if attribute == <usize>::MAX {
            return;
        }

        for branch in 0..2 {
            path.insert(item(attribute, branch));
            if let Some(cache_node) = self.cache.find(path) {
                let node_infos = self.create_solution_tree_entry(cache_node);
                let child_index = tree.add_node(index, branch == 0, TreeNode::new(node_infos));
                if !cache_node.is_leaf {
                    self.get_solution_tree_recursion(cache_node.test, path, tree, child_index)
                }
            }
            path.remove(&item(attribute, branch));
        }
    }
}

#[cfg(test)]
mod dl85_test {
    use crate::cache::trie::Trie;
    use crate::data::{BinaryData, FileReader};
    use crate::heuristics::NoHeuristic;
    use crate::searches::errors::NativeError;
    use crate::searches::optimal::dl85::DL85;
    use crate::searches::utils::{
        BranchingStrategy, CacheInitStrategy, LowerBoundStrategy, NodeExposedData, Specialization,
    };
    use crate::structures::{Bitset, RevBitset};

    #[test]
    fn run_basic_dl85() {
        let data = BinaryData::read("test_data/anneal.txt", false, 0.0);
        let mut structure = RevBitset::new(&data);
        let error_function = Box::<NativeError>::default();
        let cache = Box::<Trie>::default();
        let heuristics = Box::<NoHeuristic>::default();

        let mut learner = DL85::new(
            1,
            4,
            <f64>::INFINITY,
            600,
            false,
            0,
            CacheInitStrategy::None_,
            Specialization::Murtree,
            LowerBoundStrategy::None_,
            BranchingStrategy::None_,
            NodeExposedData::ClassesSupport,
            cache,
            error_function,
            heuristics,
        );
        learner.fit(&mut structure);
        println!("{:#?}", learner.statistics)
    }
}
