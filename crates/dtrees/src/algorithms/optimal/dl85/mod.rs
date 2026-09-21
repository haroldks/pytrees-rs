use crate::algorithms::common::errors::ErrorWrapper;
use crate::algorithms::common::heuristics::Heuristic;
use crate::algorithms::common::types::{
    BranchingChoice, BranchingPolicy, FitError, LowerBoundPolicy, NodeDataType, RuleType,
    SearchResult, SearchStatistics,
};
use crate::algorithms::optimal::depth2::{OptimalDepth2Tree};
use crate::algorithms::optimal::dl85::config::DL85Config;
use crate::algorithms::optimal::rules::common::{SimilarityLowerBoundRule, TimeLimitRule};
use crate::algorithms::optimal::rules::{
    DiscrepancyRule, GainRule, Rule, RuleContext, RuleManager,
};
use crate::algorithms::optimal::Reason;
use crate::algorithms::TreeSearchAlgorithm;
use crate::caching::{CacheEntry, CacheKey, Caching, Index, SearchPath};
use crate::cover::similarities::SimilarityCover;
use crate::cover::Cover;
use crate::globals::{attribute, float_is_null, item};
use crate::tree::{NodeInfos, Tree, TreeNode};

mod builder;
pub mod config;

pub use builder::DL85Builder;

pub struct DL85<C, D, E, H>
where
    C: Caching + ?Sized,
    D: OptimalDepth2Tree + ?Sized,
    E: ErrorWrapper + ?Sized,
    H: Heuristic + ?Sized,
{
    config: DL85Config,
    cache: Box<C>,
    error_fn: Box<E>,
    depth2_search: Box<D>,
    heuristic_fn: Box<H>,
    search_rules: RuleManager,
    node_rules: RuleManager,
    time_rule: TimeLimitRule,
    similarity_rule: SimilarityLowerBoundRule,
    statistics: SearchStatistics,
    tree: Tree,
    root_candidates: Vec<usize>,
    gain_gap: f64,
    gain_limit: f64,
}

impl<C, D, E, H> TreeSearchAlgorithm for DL85<C, D, E, H>
where
    C: Caching + ?Sized,
    D: OptimalDepth2Tree + ?Sized,
    E: ErrorWrapper + ?Sized,
    H: Heuristic + ?Sized,
{
    fn fit(&mut self, cover: &mut Cover) -> Result<(), FitError> {

        let mut result = SearchResult {
            reason: Reason::RuleReason,
            ..Default::default()
        };

        while result.reason == Reason::RuleReason && !self.time_rule.exhausted() {
            result = self.partial_fit(cover);
        }
        Ok(())
    }

    fn tree(&self) -> &Tree {
        &self.tree
    }
}

impl<C, D, E, H> DL85<C, D, E, H>
where
    C: Caching + ?Sized,
    D: OptimalDepth2Tree + ?Sized,
    E: ErrorWrapper + ?Sized,
    H: Heuristic + ?Sized,
{
    pub fn new(
        config: DL85Config,
        cache: Box<C>,
        depth2_search: Box<D>,
        error_fn: Box<E>,
        heuristic_fn: Box<H>,
        node_rules: RuleManager,
        search_rules: RuleManager,
        time_rule: TimeLimitRule,
    ) -> Self {
        Self {
            config,
            cache,
            error_fn,
            depth2_search,
            heuristic_fn,
            search_rules,
            node_rules,
            time_rule,
            similarity_rule: SimilarityLowerBoundRule::new(),
            statistics: SearchStatistics::default(),
            tree: Tree::default(),
            root_candidates: vec![],
            gain_gap: 0.0,
            gain_limit: 0.0,
        }
    }

    pub fn config(&self) -> DL85Config {
        self.config
    }

    pub fn partial_fit(&mut self, cover: &mut Cover) -> SearchResult {
        self.statistics.increment_restarts();

        let mut root_context = RuleContext::default();

        if self.statistics.restarts() <= 1 {
            self.cache.init();
            self.statistics.num_attributes = cover.num_attributes;
            self.statistics.num_samples = cover.count();

            if let Some(discrepancy_rule) = self.search_rules.get_rule_mut::<DiscrepancyRule>() {
                discrepancy_rule.update_to_true_limit(
                    self.statistics.num_attributes,
                    self.config.base.max_depth,
                );
            }

            let (error, label) = self.compute_leaf_error(cover);
            self.cache.update_root().map(|updater| {
                updater
                    .leaf_error(error)
                    .output(label)
                    .size(self.statistics.num_samples)
            });

            let mut candidates =
                self.get_candidates(cover, self.config.base.min_support, None, None);
            self.heuristic_fn.compute(cover, &mut candidates);
            self.root_candidates = candidates;

            let bound = <f64>::min(error, self.config.base.max_error);

            let branch_item = usize::MAX;
            root_context.item(branch_item);
            root_context.position(0);
            root_context.discrepancy(0);
            root_context.upper_bound(bound);
            root_context.error(bound);

            self.time_rule.activate();
            self.node_rules.activate_all();
            self.search_rules.activate_all();
            if self.config.use_similarity_lb() {
                self.similarity_rule.activate();
            }
        } else {
            root_context.upper_bound(self.statistics.tree_error);
            root_context.error(self.statistics.tree_error);
        }
        // println!("Cache size : {}", self.cache.size());
        let root_index = self.cache.root_index();
        let node_ub = self
            .cache
            .root()
            .map_or(f64::INFINITY, |node| node.upper_bound());

        root_context.node_upper_bound(node_ub);
        root_context.support(self.statistics.num_samples);

        let mut similarity = SimilarityCover::default();
        let mut search_path = SearchPath::new();
        let candidates = std::mem::take(&mut self.root_candidates);

        let mut result = self.recursive_search(
            cover,
            &mut search_path,
            &candidates,
            0,
            usize::MAX,
            root_index,
            &mut similarity,
            &mut root_context,
        );

        self.root_candidates = candidates;

        if self.statistics.restarts() <= 1 || self.gain_gap <= 0.0 {
            // println!("Min gap : {}", self.gain_gap);
            if let Some(gain_rule) = self.search_rules.get_rule_mut::<GainRule>() {
                gain_rule.update_gap_delta(self.gain_gap);
            }
        }

        if result.reason == Reason::RuleReason {
            self.node_rules.relax_all();
            self.search_rules.relax_all();
        }

        if !self.node_rules.is_active() && !self.search_rules.is_active() {
            result.reason = Reason::Done;
        }

        self.statistics.duration = self.time_rule.elapsed_seconds();
        self.statistics.tree_error = result.error;
        self.statistics.cache_size = self.cache.size();
        self.build_solution_tree();
        result
    }

    fn recursive_search(
        &mut self,
        cover: &mut Cover,
        path: &mut SearchPath,
        candidates: &[usize],
        depth: usize,
        parent_item: usize,
        parent_index: Index,
        similarity: &mut SimilarityCover,
        parent_context: &mut RuleContext,
    ) -> SearchResult {
        self.statistics.increment_search_space();

        let mut subtree_upper_bound = parent_context.upper_bound;
        let parent_key = parent_index.to_cache_key(path);
        let result = self.evaluate(parent_context, &parent_key, RuleType::Node);

        if !result.0 {
            // println!("Parent context: {:?} Result ? : {:?}", parent_context.gain, result);
            return SearchResult {
                error: result.2,
                has_intersected: false,
                reason: result.1,
            };
        }

        if !parent_index.is_new() {
            cover.branch_on(parent_item);
        }

        if self.config.use_similarity_lb() {
            let similarity = similarity.compute_similarity(cover.sparse());
            parent_context.node_lower_bound = similarity.max(parent_context.node_lower_bound);
            let result = self.evaluate_node(parent_context, &parent_key, RuleType::Similarity);
            if !result.0 {
                return SearchResult {
                    error: result.2,
                    has_intersected: true,
                    reason: result.1,
                };
            }
        }

        let mut node_candidates = self.get_candidates(
            cover,
            self.config.base.min_support,
            Some(&candidates),
            Some(attribute(parent_item)),
        );

        if node_candidates.is_empty() {
            let error = self
                .cache
                .update_node(&parent_key)
                .map_or(f64::INFINITY, |updater| updater.leaf().get_error());
            return SearchResult {
                error,
                has_intersected: true,
                reason: Reason::NoCandidates,
            };
        }

        if self.config.use_depth2_optimization() && self.config.base.max_depth - depth <= 2 {
            // println!("Size : {}", self.cache.size());
            let result = self.apply_specialized_depth2_search(
                cover,
                &node_candidates,
                parent_index,
                subtree_upper_bound,
                path,
                self.config.base.max_depth - depth,
            );
            // println!("Size : {}", self.cache.size());
            // self.cache.print();

            match result {
                Err(_) => {}
                Ok(search_result) => return search_result,
            }
        }

        let mut scores = vec![];
        if self.config.always_sort {
            scores = self.heuristic_fn.compute(cover, &mut node_candidates);
        }

        let mut subtree_similarity_data = SimilarityCover::default();
        let mut min_lower_bound = <f64>::INFINITY;

        let mut rule_pruned = false;
        for (position, &child) in node_candidates.iter().enumerate() {
            let mut branch_context = RuleContext::default();
            branch_context.discrepancy(parent_context.discrepancy + position);
            branch_context.position(position);

            if scores.len() > 1 {
                branch_context.gain(parent_context.gain + (scores[0] - scores[position]));
                if (self.statistics.restarts() <= 1 || self.gain_gap <= 0.0) && (self.gain_gap <= 0f64 || branch_context.gain < self.gain_gap) {
                    self.gain_gap = branch_context.gain;
                }
            }

            let search_result = self.evaluate_node(&branch_context, &parent_key, RuleType::Search);

            if !search_result.0 {
                return SearchResult {
                    error: search_result.2,
                    has_intersected: true,
                    reason: search_result.1,
                };
            }

            let (first_branch, first_lb, second_lb) =
                self.determine_branch_strategy(child, path, cover, &subtree_similarity_data);
            branch_context.depth(depth + 1);
            let branch_item = item(child, first_branch);
            branch_context.item(branch_item);
            branch_context.node_lower_bound(first_lb);
            branch_context.upper_bound(subtree_upper_bound);

            let (first_result, branch_key) = self.process_branch(
                cover,
                path,
                &mut branch_context,
                &mut subtree_similarity_data,
                &node_candidates,
                depth,
            );

            if first_result.error >= subtree_upper_bound - second_lb {
                min_lower_bound = self
                    .cache
                    .node(&branch_key)
                    .map_or(min_lower_bound, |node| {
                        let stored_lb = match first_result.error.is_finite() {
                            true => first_result.error + second_lb,
                            false => node.lower_bound() + second_lb,
                        };
                        stored_lb.min(min_lower_bound)
                    });
                self.statistics.increment_sibling_pruning();
                continue;
            }

            let mut branch_context = RuleContext::default();
            let right_ub = subtree_upper_bound - first_result.error;
            let branch_item = item(child, 1 - first_branch);
            branch_context.item(branch_item);
            branch_context.depth(depth + 1);
            branch_context.position(position);
            branch_context.discrepancy(parent_context.discrepancy + position);
            branch_context.node_lower_bound(second_lb);
            branch_context.upper_bound(right_ub);

            let (second_result, _) = self.process_branch(
                cover,
                path,
                &mut branch_context,
                &mut subtree_similarity_data,
                &node_candidates,
                depth,
            );

            rule_pruned |= first_result.reason == Reason::RuleReason
                || second_result.reason == Reason::RuleReason;

            let subtree_error = first_result.error + second_result.error;
            if subtree_error < subtree_upper_bound {
                subtree_upper_bound = subtree_error;
                let optimal = self
                    .cache
                    .update_node(&parent_key)
                    .map_or(false, |mut updater| {
                        updater = updater.error(subtree_error).test(child);

                        if float_is_null(updater.get_lower_bound() - subtree_error) {
                            updater.upper_bound(parent_context.upper_bound).optimal();
                            return true;
                        }
                        false
                    });

                if optimal {
                    return SearchResult {
                        error: subtree_error,
                        has_intersected: true,
                        reason: Reason::Done,
                    };
                }
            } else {
                min_lower_bound = min_lower_bound.min(subtree_error);
            }
        }

        let error = self
            .cache
            .update_node(&parent_key)
            .map_or(f64::INFINITY, |mut updater| {
                if rule_pruned {
                    updater = updater.upper_bound(f64::INFINITY);
                } else {
                    updater = updater.optimal().upper_bound(parent_context.upper_bound);
                }
                let error = updater.get_error();
                if error.is_infinite() {
                    let lb = updater
                        .get_lower_bound()
                        .max(min_lower_bound.max(parent_context.upper_bound));
                    updater.lower_bound(lb);
                }
                error
            });

        SearchResult {
            error,
            has_intersected: true,
            reason: if rule_pruned {
                Reason::RuleReason
            } else {
                Reason::Done
            },
        }
    }

    fn process_branch(
        &mut self,
        cover: &mut Cover,
        path: &mut SearchPath,
        branch_context: &mut RuleContext,
        similarity_cover: &mut SimilarityCover,
        candidates: &[usize],
        current_depth: usize,
    ) -> (SearchResult, CacheKey) {
        path.push(branch_context.item);
        let branch_key_vec = path.to_sorted_vec();
        let branch_index = self.cache.insert(&branch_key_vec);
        let branch_key = branch_index.to_cache_key(path);

        if branch_index.is_new() {
            let size = cover.branch_on(branch_context.item);
            branch_context.support(size);
            let error = self.compute_leaf_error(cover);
            // branch_context.error(error.0);
            branch_context.leaf_error(error.0);
            branch_context.node_upper_bound(f64::INFINITY);

            self.cache.update_node(&branch_key).map(|updater| {
                updater
                    .leaf_error(error.0)
                    .output(error.1)
                    .lower_bound(branch_context.node_lower_bound) // TODO
                    .size(size)
            });
        } else {
            self.statistics.increment_cache_hits();
            if let Some(node) = self.cache.node(&branch_key) {
                // branch_context.error(node.error().min(node.leaf_error())); // TODO
                branch_context.error(node.error());
                branch_context.support(node.size());
                branch_context.node_upper_bound(node.upper_bound());
                branch_context.leaf_error(node.leaf_error())
            }
        }
        let first_result = self.recursive_search(
            cover,
            path,
            candidates,
            current_depth + 1,
            branch_context.item,
            branch_index,
            similarity_cover,
            branch_context,
        );

        self.backtrack(
            cover,
            path,
            branch_index,
            &branch_context.item,
            &first_result,
            similarity_cover,
        );

        (first_result, branch_key)
    }

    fn evaluate(
        &mut self,
        context: &RuleContext,
        key: &CacheKey,
        rule_type: RuleType,
    ) -> (bool, Reason, f64) {
        let time_result = self.evaluate_node(context, key, RuleType::Time);
        if !time_result.0 {
            return time_result;
        }
        self.evaluate_node(context, key, rule_type)
    }

    fn evaluate_node(
        &mut self,
        context: &RuleContext,
        key: &CacheKey,
        rule_type: RuleType,
    ) -> (bool, Reason, f64) {
        let result = match rule_type {
            RuleType::Node => self.node_rules.evaluate(context),
            RuleType::Search => self.search_rules.evaluate(context),
            RuleType::Time => self.time_rule.evaluate(context),
            RuleType::Similarity => self.similarity_rule.evaluate(context),
        };

        let mut error = f64::INFINITY;
        if let Some(mut updater) = self.cache.update_node(key) {
            if let Some(bound) = result.modified_bound {
                updater = updater.upper_bound(bound);
            }

            if result.optimal.unwrap_or(false) {
                updater = updater.optimal();
            }

            if result.leaf.unwrap_or(false) {
                // println!("Error : {:?}", updater.get_error());
                updater = updater.leaf();
                // println!("Error : {:?}", updater.get_error());
            }

            if rule_type == RuleType::Similarity {
                updater = updater.lower_bound(context.node_lower_bound)
            }
            // println!("Error : {:?}", updater.get_error());
            error = updater.get_error().min(updater.get_leaf_error());
            // println!("Error : {:?}", updater.get_error());
        }

        (result.continue_search, result.reason, error)
    }

    fn compute_leaf_error(&self, cover: &mut Cover) -> (f64, f64) {
        if self.config.data_type == NodeDataType::ClassesSupport {
            return self.error_fn.compute(&cover.labels_count());
        }
        self.error_fn.compute(&cover.to_vec())
    }

    pub fn statistics(&self) -> &SearchStatistics {
        &self.statistics
    }

    pub fn elapsed_seconds(&self) -> f64 {
        self.time_rule.elapsed_seconds()
    }

    pub fn time_is_exhausted(&self) -> bool {
        self.time_rule.exhausted()
    }

    fn determine_branch_strategy(
        &self,
        attribute: usize,
        path: &mut SearchPath,
        cover: &mut Cover,
        similarity: &SimilarityCover,
    ) -> BranchingChoice {
        let mut branch_first = 0;
        let mut bounds = [0.0, 0.0];

        match self.config.branching_policy {
            BranchingPolicy::Default => {}
            BranchingPolicy::Dynamic => {
                bounds = self.get_cached_branch_bounds(attribute, path);
                if let LowerBoundPolicy::Similarity = self.config.lower_bound_policy {
                    self.enhance_bounds_with_similarity(&mut bounds, attribute, cover, similarity);
                }
                branch_first = (bounds[1] > bounds[0]) as usize;
            }
        }

        let first_bound = bounds[branch_first];
        let second_bound = bounds[1 - branch_first];

        (branch_first, first_bound, second_bound)
    }

    fn get_cached_branch_bounds(&self, attribute: usize, path: &mut SearchPath) -> [f64; 2] {
        let mut bounds = [0.0; 2];
        for (branch, lb) in bounds.iter_mut().enumerate() {
            let branch_item = item(attribute, branch);
            path.push(branch_item);
            let key = path.to_key();
            if let Some(node) = self.cache.node(&key) {
                let error = node.error();
                *lb = match error.is_finite() {
                    // TODO: Investigate
                    true => error,
                    false => node.lower_bound(),
                }
            }
            path.remove(&branch_item)
        }
        bounds
    }

    fn enhance_bounds_with_similarity(
        &self,
        bounds: &mut [f64; 2],
        attribute: usize,
        cover: &mut Cover,
        similarity: &SimilarityCover,
    ) {
        for (branch, lb) in bounds.iter_mut().enumerate() {
            let branch_item = item(attribute, branch);
            cover.branch_on(branch_item);
            let similarity_lb = similarity.compute_similarity(cover.sparse());
            *lb = lb.max(similarity_lb);
            cover.backtrack()
        }
    }

    fn backtrack(
        &mut self,
        cover: &mut Cover,
        path: &mut SearchPath,
        index: Index,
        item: &usize,
        search_result: &SearchResult,
        similarity: &mut SimilarityCover,
    ) {
        if !(index.is_new() || search_result.has_intersected) {
            cover.branch_on(*item);
        }

        // Update similarity data if applicable
        if self.config.use_similarity_lb() && search_result.reason == Reason::LowerBoundConstrained
        {
            let key = index.to_cache_key(path);
            if let Some(node) = self.cache.node(&key) {
                similarity.update(cover.sparse(), node.error())
            }
        }
        cover.backtrack();
        path.remove(item);
    }

    fn apply_specialized_depth2_search(
        &mut self,
        cover: &mut Cover,
        _candidates: &[usize],
        parent_index: Index,
        upper_bound: f64,
        path: &mut SearchPath,
        depth: usize,
    ) -> Result<SearchResult, FitError> {
        let key = parent_index.to_cache_key(path);
        if let Some(node) = self.cache.node(&key) {
            if upper_bound < node.lower_bound() {
                return Ok(SearchResult {
                    error: node.error(),
                    has_intersected: true,
                    reason: Reason::LowerBoundConstrained,
                });
            }
        }
        let tree_result = self
            .depth2_search
            .fit(self.config.base.min_support, depth, cover, None);
        match tree_result {
            Err(err) => Err(err),
            Ok(tree) => {
                // tree.print();
                // println!("{:?}", parent_index);
                let error = tree.root_error();
                self.cache_specialized_depth2_tree_results(
                    path,
                    parent_index,
                    &tree,
                    tree.get_root_index(),
                );

                Ok(SearchResult {
                    error,
                    has_intersected: true,
                    reason: Reason::FromSpecializedAlgorithm,
                })
            }
        }
    }

    fn cache_specialized_depth2_tree_results(
        &mut self,
        path: &mut SearchPath,
        parent_index: Index,
        tree: &Tree,
        tree_index: usize,
    ) {
        let parent_key = parent_index.to_cache_key(path);
        let node_test = tree.node_test(tree_index);
        if let Some(mut updater) = self.cache.update_node(&parent_key) {
            updater = updater
                .error(tree.node_error(tree_index))
                .leaf_error(tree.node_error(tree_index))
                .upper_bound(tree.node_error(tree_index))
                .optimal();

            if tree.node_test(tree_index).is_none() {
                updater
                    .leaf()
                    .output(tree.node_output(tree_index).unwrap_or(0.0));
                return;
            }
            updater.test(node_test.unwrap());
        }

        let node_test = node_test.unwrap();

        let children = tree.node_children(tree_index);
        let children = [children.0, children.1];
        for (branch, &tree_branch_index) in children.iter().enumerate() {
            if tree_branch_index > 0 {
                let branch_item = item(node_test, branch);
                path.push(branch_item);
                let branch_key_vec = path.to_sorted_vec();
                let cache_branch_index = self.cache.insert(&branch_key_vec);
                self.cache_specialized_depth2_tree_results(
                    path,
                    cache_branch_index,
                    tree,
                    tree_branch_index,
                );
                path.remove(&branch_item);
            }
        }
    }

    pub fn cache_entry_to_tree_entry(&self, cache_entry: &CacheEntry) -> NodeInfos {
        NodeInfos {
            error: cache_entry.error(),
            out: if cache_entry.is_leaf() {
                Some(cache_entry.out())
            } else {
                None
            },
            test: if cache_entry.is_leaf() {
                None
            } else {
                Some(cache_entry.test())
            },
            metric: None,
        }
    }

    fn build_solution_tree(&mut self) {
        let mut tree = Tree::default();
        let mut path = SearchPath::new();
        if let Some(cache_root) = self.cache.root() {
            let tree_entry = self.cache_entry_to_tree_entry(cache_root);
            let root = tree.add_root(TreeNode::new(tree_entry));
            self.build_tree_branches(cache_root.test(), &mut path, &mut tree, root);
        }
        self.tree = tree;
    }

    fn build_tree_branches(
        &self,
        attribute: usize,
        path: &mut SearchPath,
        tree: &mut Tree,
        index: usize,
    ) {
        if attribute == usize::MAX {
            return;
        }

        for branch in 0..2 {
            let branch_item = item(attribute, branch);
            path.push(branch_item);
            let key = path.to_key();
            if let Some(node) = self.cache.node(&key) {
                let branch_entry = self.cache_entry_to_tree_entry(node);
                let child_index = tree.add_node(index, branch == 0, TreeNode::new(branch_entry));
                if !node.is_leaf() {
                    self.build_tree_branches(node.test(), path, tree, child_index);
                }
            }
            path.remove(&branch_item);
        }
    }
}

#[cfg(test)]
mod dl85_test {
    use crate::algorithms::common::errors::NativeError;
    use crate::algorithms::common::heuristics::NoHeuristic;
    use crate::algorithms::common::types::OptimalDepth2Policy;
    use crate::algorithms::optimal::depth2::ErrorMinimizer;
    use crate::algorithms::optimal::dl85::{DL85Builder, DL85};
    use crate::algorithms::TreeSearchAlgorithm;
    use crate::caching::Trie;
    use crate::reader::data_reader::DataReader;
    use std::path::Path;

    #[test]
    fn run() -> Result<(), Box<dyn std::error::Error>> {
        let reader = DataReader::default();
        let path = Path::new("test_data/anneal.txt");
        let mut cover = reader.read_file(path)?;

        let error_fn = Box::<NativeError>::default();

        let depth2 = Box::new(ErrorMinimizer::new(error_fn.clone()));

        let mut algo = DL85Builder::default()
            .max_depth(2)
            .min_support(50)
            .max_time(10.0)
            .specialization(OptimalDepth2Policy::Enabled)
            .cache(Box::<Trie>::default())
            .heuristic(Box::<NoHeuristic>::default())
            .depth2_search(depth2)
            .error_function(error_fn)
            .build()?;
        // Configure and build the DL85 algorithm using builder pattern

        // Execute the fitting process and handle any errors
        algo.fit(&mut cover)?;

        // Report results
        println!("Search statistics: {:#?}", algo.statistics);
        println!("Execution time: {:.3}s", algo.time_rule.elapsed_seconds());

        // Print resulting tree
        algo.tree.print();

        Ok(())
    }
}
