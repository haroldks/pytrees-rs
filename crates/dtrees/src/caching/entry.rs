#[derive(Copy, Clone, Debug)]
pub struct CacheEntry {
    item: usize,
    test: usize,
    error: f64,
    upper_bound: f64,
    lower_bound: f64,
    metric: f64,
    size: usize,
    leaf_error: f64,
    out: f64,
    is_optimal: bool,
    is_leaf: bool,
}
impl CacheEntry {
    pub fn new(item: usize) -> Self {
        Self {
            item,
            test: <usize>::MAX,
            error: f64::INFINITY,
            upper_bound: f64::INFINITY,
            lower_bound: 0.0,
            metric: 0.0,
            size: 0,
            leaf_error: f64::INFINITY,
            out: 0.0,
            is_optimal: false,
            is_leaf: false,
        }
    }

    pub fn item(&self) -> usize {
        self.item
    }

    pub fn test(&self) -> usize {
        self.test
    }

    pub fn error(&self) -> f64 {
        self.error
    }

    pub fn upper_bound(&self) -> f64 {
        self.upper_bound
    }

    pub fn lower_bound(&self) -> f64 {
        self.lower_bound
    }

    pub fn metric(&self) -> f64 {
        self.metric
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn leaf_error(&self) -> f64 {
        self.leaf_error
    }

    pub fn out(&self) -> f64 {
        self.out
    }

    pub fn is_optimal(&self) -> bool {
        self.is_optimal
    }

    pub fn is_leaf(&self) -> bool {
        self.is_leaf
    }

    pub fn has_valid_test(&self) -> bool {
        self.test != usize::MAX
    }

    pub fn has_finite_error(&self) -> bool {
        self.error.is_finite()
    }

    pub fn has_finite_upper_bound(&self) -> bool {
        self.upper_bound.is_finite()
    }

    pub fn has_finite_leaf_error(&self) -> bool {
        self.leaf_error.is_finite()
    }
}

impl Default for CacheEntry {
    fn default() -> Self {
        Self {
            item: <usize>::MAX,
            test: <usize>::MAX,
            error: f64::INFINITY,
            upper_bound: f64::INFINITY,
            lower_bound: 0.0,
            metric: 0.0,
            size: 0,
            leaf_error: f64::INFINITY,
            out: 0.0,
            is_optimal: false,
            is_leaf: false,
        }
    }
}

pub struct CacheEntryUpdater<'a> {
    node: &'a mut CacheEntry,
}

impl<'a> CacheEntryUpdater<'a> {
    pub fn new(node: &'a mut CacheEntry) -> Self {
        Self { node }
    }

    pub fn item(self, item: usize) -> Self {
        self.node.item = item;
        self
    }

    pub fn test(self, test: usize) -> Self {
        self.node.test = test;
        self
    }

    pub fn error(self, error: f64) -> Self {
        self.node.error = error;
        self
    }

    pub fn upper_bound(self, upper_bound: f64) -> Self {
        self.node.upper_bound = upper_bound;
        self
    }
    pub fn lower_bound(self, lower_bound: f64) -> Self {
        self.node.lower_bound = lower_bound;
        self
    }

    pub fn metric(self, metric: f64) -> Self {
        self.node.metric = metric;
        self
    }

    pub fn size(self, size: usize) -> Self {
        self.node.size = size;
        self
    }

    pub fn leaf_error(self, leaf_error: f64) -> Self {
        self.node.leaf_error = leaf_error;
        self
    }

    pub fn output(self, output: f64) -> Self {
        self.node.out = output;
        self
    }

    pub fn optimal(self) -> Self {
        self.node.is_optimal = true;
        self
    }

    pub fn leaf(self) -> Self {
        self.node.is_leaf = true;
        self.node.error = self.node.leaf_error;
        self
    }

    pub fn get_error(&self) -> f64 {
        self.node.error
    }

    pub fn get_leaf_error(&self) -> f64 {
        self.node.leaf_error
    }

    pub fn get_lower_bound(&self) -> f64 {
        self.node.lower_bound
    }
}
