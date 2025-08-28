use std::any::Any;
use std::time::Duration;

// For downcasting support
pub trait AsAny {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// Core context trait
pub trait BaseContext: AsAny {
    // Core methods every context must implement
    fn support(&self) -> usize;
    fn min_sup(&self) -> usize;
    fn current_depth(&self) -> usize;
    fn max_depth(&self) -> usize;
    fn current_time(&self) -> Duration;
    fn max_time(&self) -> f64;
    fn upper_bound(&self) -> f64;
}

// Discrepancy context trait
pub trait DiscrepancyContext: BaseContext {
    fn discrepancy(&self) -> usize;
    fn discrepancy_budget(&self) -> usize;
}

// Purity context trait
pub trait PurityContext: BaseContext {
    fn purity_threshold(&self) -> f64;
}

// Beam context struct
pub struct BeamContext {
    support: usize,
    min_sup: usize,
    depth: usize,
    max_depth: usize,
    current_time: Duration,
    max_time: f64,
    upper_bound: f64,

    discrepancy: Option<usize>,
    discrepancy_budget: Option<usize>,
    similarity_bound: Option<f64>,
    candidates_count: Option<usize>,
    position: Option<usize>,
    branch_budget: Option<usize>,
    purity_threshold: Option<f64>,
}

impl BeamContext {
    pub fn new(
        support: usize,
        min_sup: usize,
        depth: usize,
        max_depth: usize,
        current_time: Duration,
        max_time: f64,
        upper_bound: f64,
    ) -> Self {
        Self {
            support,
            min_sup,
            depth,
            max_depth,
            current_time,
            max_time,
            upper_bound,
            discrepancy: None,
            discrepancy_budget: None,
            similarity_bound: None,
            candidates_count: None,
            position: None,
            branch_budget: None,
            purity_threshold: None,
        }
    }

    // TODO : Maybe use references
    pub fn with_discrepancy(mut self, discrepancy: usize, budget: usize) -> Self {
        self.discrepancy = Some(discrepancy);
        self.discrepancy_budget = Some(budget);
        self
    }

    pub fn with_similarity(mut self, bound: f64) -> Self {
        self.similarity_bound = Some(bound);
        self
    }

    pub fn with_candidates(mut self, count: usize) -> Self {
        self.candidates_count = Some(count);
        self
    }

    pub fn with_position(mut self, position: usize, budget: usize) -> Self {
        self.position = Some(position);
        self.branch_budget = Some(budget);
        self
    }

    pub fn with_purity_threshold(mut self, threshold: f64) -> Self {
        self.purity_threshold = Some(threshold);
        self
    }
}

impl AsAny for BeamContext {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl BaseContext for BeamContext {
    fn support(&self) -> usize {
        self.support
    }

    fn min_sup(&self) -> usize {
        self.min_sup
    }

    fn current_depth(&self) -> usize {
        self.depth
    }

    fn max_depth(&self) -> usize {
        self.max_depth
    }

    fn current_time(&self) -> Duration {
        self.current_time
    }

    fn max_time(&self) -> f64 {
        self.max_time
    }

    fn upper_bound(&self) -> f64 {
        self.upper_bound
    }
}

impl DiscrepancyContext for BeamContext {
    fn discrepancy(&self) -> usize {
        self.discrepancy
            .expect("Discrepancy is not set in the context")
    }

    fn discrepancy_budget(&self) -> usize {
        self.discrepancy_budget
            .expect("Discrepancy budget is not set in the context")
    }
}

impl PurityContext for BeamContext {
    fn purity_threshold(&self) -> f64 {
        self.purity_threshold
            .expect("Purity is not set in the context")
    }
}

// Implement BaseContext for boxed trait objects
impl AsAny for Box<dyn BaseContext> {
    fn as_any(&self) -> &dyn Any {
        (**self).as_any()
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        (**self).as_any_mut()
    }
}

impl BaseContext for Box<dyn BaseContext> {
    fn support(&self) -> usize {
        (**self).support()
    }

    fn min_sup(&self) -> usize {
        (**self).min_sup()
    }

    fn current_depth(&self) -> usize {
        (**self).current_depth()
    }

    fn max_depth(&self) -> usize {
        (**self).max_depth()
    }

    fn current_time(&self) -> Duration {
        (**self).current_time()
    }

    fn max_time(&self) -> f64 {
        (**self).max_time()
    }

    fn upper_bound(&self) -> f64 {
        (**self).upper_bound()
    }
}

// Similar implementations for other context types
impl AsAny for Box<dyn DiscrepancyContext> {
    fn as_any(&self) -> &dyn Any {
        (**self).as_any()
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        (**self).as_any_mut()
    }
}

impl BaseContext for Box<dyn DiscrepancyContext> {
    fn support(&self) -> usize {
        (**self).support()
    }

    fn min_sup(&self) -> usize {
        (**self).min_sup()
    }

    fn current_depth(&self) -> usize {
        (**self).current_depth()
    }

    fn max_depth(&self) -> usize {
        (**self).max_depth()
    }

    fn current_time(&self) -> Duration {
        (**self).current_time()
    }

    fn max_time(&self) -> f64 {
        (**self).max_time()
    }

    fn upper_bound(&self) -> f64 {
        (**self).upper_bound()
    }
}

impl DiscrepancyContext for Box<dyn DiscrepancyContext> {
    fn discrepancy(&self) -> usize {
        (**self).discrepancy()
    }

    fn discrepancy_budget(&self) -> usize {
        (**self).discrepancy_budget()
    }
}

// PurityContext implementations
impl AsAny for Box<dyn PurityContext> {
    fn as_any(&self) -> &dyn Any {
        (**self).as_any()
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        (**self).as_any_mut()
    }
}

impl BaseContext for Box<dyn PurityContext> {
    fn support(&self) -> usize {
        (**self).support()
    }

    fn min_sup(&self) -> usize {
        (**self).min_sup()
    }

    fn current_depth(&self) -> usize {
        (**self).current_depth()
    }

    fn max_depth(&self) -> usize {
        (**self).max_depth()
    }

    fn current_time(&self) -> Duration {
        (**self).current_time()
    }

    fn max_time(&self) -> f64 {
        (**self).max_time()
    }

    fn upper_bound(&self) -> f64 {
        (**self).upper_bound()
    }
}

impl PurityContext for Box<dyn PurityContext> {
    fn purity_threshold(&self) -> f64 {
        (**self).purity_threshold()
    }
}
