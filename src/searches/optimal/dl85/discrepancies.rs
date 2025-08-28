pub trait Discrepancy {
    fn next(&mut self) -> usize;
}

pub struct MonotonicDiscrepancy {
    increment: usize,
    current: usize,
}

impl Default for MonotonicDiscrepancy {
    fn default() -> Self {
        Self {
            increment: 1,
            current: 0,
        }
    }
}

impl Discrepancy for MonotonicDiscrepancy {
    fn next(&mut self) -> usize {
        let value = self.current;
        self.current += self.increment;
        value
    }
}

impl MonotonicDiscrepancy {
    pub fn new(increment: usize) -> Self {
        Self {
            current: 0,
            increment,
        }
    }
}

pub struct ExponentialDiscrepancy {
    current: usize,
    base: usize,
}

impl Default for ExponentialDiscrepancy {
    fn default() -> Self {
        Self {
            current: 1,
            base: 2,
        }
    }
}

impl ExponentialDiscrepancy {
    pub fn new(base: usize) -> Self {
        Self { current: 1, base }
    }
}

impl Discrepancy for ExponentialDiscrepancy {
    fn next(&mut self) -> usize {
        let value = self.current;
        self.current *= self.base;
        value
    }
}

pub struct LubyDiscrepancy {
    multiplier: usize,
    steps: Vec<usize>,
    current: usize,
    iter: usize,
}

impl Default for LubyDiscrepancy {
    fn default() -> Self {
        Self {
            multiplier: 1,
            steps: vec![1],
            current: 1,
            iter: 1,
        }
    }
}

impl LubyDiscrepancy {
    pub fn new(multiplier: usize) -> Self {
        Self {
            multiplier,
            steps: vec![1],
            current: multiplier,
            iter: 1,
        }
    }
}

impl Discrepancy for LubyDiscrepancy {
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

#[cfg(test)]
mod discrepancy_test {
    use crate::searches::optimal::dl85::discrepancies::{
        Discrepancy, ExponentialDiscrepancy, LubyDiscrepancy, MonotonicDiscrepancy,
    };

    #[test]
    fn test_monotonic() {
        for i in 1..5 {
            let mut monotonic = MonotonicDiscrepancy::new(i);
            let mut value = 0;
            for _ in 0..10 {
                let x = monotonic.next();
                assert_eq!(x, value);
                value += i;
            }
        }
    }

    #[test]
    fn test_exponential() {
        for i in 2..4 {
            let mut exponential = ExponentialDiscrepancy::new(i);
            let mut value = 1;
            for _ in 0..10 {
                let x = exponential.next();
                assert_eq!(x, value);
                value *= i;
            }
        }
    }

    #[test]
    fn test_luby() {
        let mut luby = LubyDiscrepancy::default();
        for _ in 0..60 {
            let x = luby.next();
            print!("{x} ")
        }
        println!("Increments : {:?}", luby.steps)
    }
}
