/// Controls how the threshold count grows between DL8.5 restarts in `MIDSearch`.
///
/// Mirrors the `StepStrategy` pattern used by `DiscrepancyRule` and `GainRule`.
#[derive(Clone, Debug)]
pub struct ThresholdSchedule {
    pub initial_n: usize,
    pub increment: usize,
    /// Maximum thresholds to use. 0 means "use however many MID computed".
    pub limit: usize,
    current_n: usize,
}

impl ThresholdSchedule {
    pub fn new(initial_n: usize, increment: usize, limit: usize) -> Self {
        Self {
            initial_n,
            increment,
            limit,
            current_n: initial_n,
        }
    }

    /// Current N (number of binary features for this iteration).
    pub fn current(&self) -> usize {
        self.current_n
    }

    /// Advance to the next threshold count.
    pub fn step(&mut self) {
        self.current_n = self.current_n.saturating_add(self.increment);
    }

    /// Returns true when no further iterations should be run.
    /// `max_available` is the total number of thresholds computed by MID.
    pub fn exhausted(&self, max_available: usize) -> bool {
        if self.current_n > max_available {
            return true;
        }
        if self.limit > 0 && self.current_n > self.limit {
            return true;
        }
        false
    }

    /// Reset back to the initial value.
    pub fn reset(&mut self) {
        self.current_n = self.initial_n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_steps_correctly() {
        let mut sched = ThresholdSchedule::new(5, 2, 11);
        assert_eq!(sched.current(), 5);
        sched.step();
        assert_eq!(sched.current(), 7);
        sched.step();
        assert_eq!(sched.current(), 9);
        sched.step();
        assert_eq!(sched.current(), 11);
        sched.step();
        assert_eq!(sched.current(), 13);
        assert!(sched.exhausted(100), "exceeded limit");
    }

    #[test]
    fn exhausted_by_max_available() {
        let mut sched = ThresholdSchedule::new(5, 1, 0); // no limit
        sched.current_n = 10;
        assert!(sched.exhausted(9));
        assert!(!sched.exhausted(10));
    }
}
