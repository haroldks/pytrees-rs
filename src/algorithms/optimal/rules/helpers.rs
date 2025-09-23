pub trait StepStrategy: Send + Sync {
    fn next(&mut self) -> usize;
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
}
