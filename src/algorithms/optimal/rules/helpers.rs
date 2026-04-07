pub trait StepStrategy: Send + Sync {
    fn next(&mut self) -> usize;
    fn estimate_steps(&self, limit: usize) -> usize;
}

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

    fn estimate_steps(&self, limit: usize) -> usize {
        if self.increment == 0 {
            return usize::MAX;
        }
        limit.saturating_add(self.increment - 1) / self.increment
    }
}

impl Monotonic {
    pub fn new(increment: usize) -> Self {
        Self {
            current: 0,
            increment,
        }
    }
}

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

    fn estimate_steps(&self, limit: usize) -> usize {
        if limit <= 1 {
            return 1;
        }
        if self.base <= 1 {
            return usize::MAX;
        }
        (limit as f64).log(self.base as f64).ceil() as usize
    }
}

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

    fn estimate_steps(&self, limit: usize) -> usize {
        let normalized_limit = limit / self.multiplier;
        if normalized_limit == 0 {
            return 1;
        }
        let log_l = (normalized_limit as f64).log2();
        if log_l < 1.0 {
            return 1;
        }

        (2.0 * normalized_limit as f64 / log_l).ceil() as usize
    }
}
