/// A sequence of budgets used to relax a rule between passes.
pub trait StepStrategy: Send + Sync {
    /// The next value of the sequence.
    fn next(&mut self) -> usize;
}

/// `0, n, 2n, 3n, …`
pub struct Monotonic {
    increment: usize,
    current: usize,
}

impl Default for Monotonic {
    fn default() -> Self {
        Self {
            increment: 1,
            current: 0,
        }
    }
}

impl StepStrategy for Monotonic {
    fn next(&mut self) -> usize {
        let value = self.current;
        self.current += self.increment;
        value
    }
}

impl Monotonic {
    /// A sequence with step `increment`.
    pub fn new(increment: usize) -> Self {
        Self {
            current: 0,
            increment,
        }
    }
}

/// `1, b, b², b³, …`
pub struct Exponential {
    current: usize,
    base: usize,
}

impl Default for Exponential {
    fn default() -> Self {
        Self {
            current: 1,
            base: 2,
        }
    }
}

impl Exponential {
    /// A sequence with base `base`.
    pub fn new(base: usize) -> Self {
        Self { current: 1, base }
    }
}

impl StepStrategy for Exponential {
    fn next(&mut self) -> usize {
        let value = self.current;
        self.current *= self.base;
        value
    }
}

/// Running sums of the Luby sequence `1, 1, 2, 1, 1, 2, 4, …`, scaled by a
/// multiplier, as used for restarts in SAT solvers.
pub struct Luby {
    multiplier: usize,
    steps: Vec<usize>,
    current: usize,
    iter: usize,
}

impl Default for Luby {
    fn default() -> Self {
        Self {
            multiplier: 1,
            steps: vec![1],
            current: 1,
            iter: 1,
        }
    }
}

impl Luby {
    /// A sequence scaled by `multiplier`.
    pub fn new(multiplier: usize) -> Self {
        Self {
            multiplier,
            steps: vec![1],
            current: multiplier,
            iter: 1,
        }
    }
}

impl StepStrategy for Luby {
    fn next(&mut self) -> usize {
        let value = self.current;
        self.iter += 1;
        let increment = match (self.iter + 1).is_power_of_two() {
            true => 2_usize.pow((self.iter + 1).ilog2() - 1),
            false => {
                let index = self.iter - 2_usize.pow(self.iter.ilog2());
                self.steps[index]
            }
        };
        self.steps.push(increment);
        self.current += increment * self.multiplier;
        value
    }
}
