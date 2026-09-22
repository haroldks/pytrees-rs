//! How the anytime search widens its budget from one pass to the next.
//!
//! Each pass of `ConTreeLds` runs under a [`Budget`]: how far down the Gini
//! ranking of features it may stray (the discrepancy), and how many of the
//! best-ranked splits it may try at each node. When a pass is cut short by its
//! budget, the search asks its [`BudgetSchedule`] for the next one.
//!
//! The schedule is told what the last pass did ([`PassReport`]) — whether the
//! discrepancy or the split budget actually cut anything — so schedules that
//! react to the search can be written without touching the solver. The fixed
//! schedules ignore it.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// The limits one pass of the anytime search runs under.
///
/// Both are counted as discrepancies, from 0: a discrepancy of 0 means "only
/// the heuristic's first choice", and each unit allows one step further down
/// its ranking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    /// How far down the Gini ranking of features the search may go,
    /// accumulated along the path from the root.
    pub discrepancy: usize,
    /// How far down the Gini ranking of split points each node may go: the
    /// node tries its best `split_discrepancy + 1` splits.
    pub split_discrepancy: usize,
}

impl Budget {
    /// The number of best-ranked splits a node may try under this budget.
    pub fn split_budget(&self) -> usize {
        self.split_discrepancy + 1
    }
}

/// What a finished pass tells the schedule.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PassReport {
    /// The pass found a better tree than any earlier pass.
    pub improved: bool,
    /// The discrepancy budget kept the pass from trying some feature.
    pub cut_by_discrepancy: bool,
    /// The split budget kept the pass from trying some split point.
    pub cut_by_split: bool,
}

/// The largest budget that can matter on a given problem. A pass at this
/// budget is the unrestricted search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScheduleBounds {
    /// Largest useful feature discrepancy for this depth and feature count.
    pub max_discrepancy: usize,
    /// Most candidate splits any feature has at the root.
    pub max_splits: usize,
    /// Whether the split budget constrains the search at all. With the `mid`
    /// split selector it applies only to the first pass, so a schedule that
    /// varies it would produce passes that are otherwise identical.
    pub split_applies: bool,
}

/// Produces the budget of each pass.
pub trait BudgetSchedule: Send {
    /// The budget of the first pass.
    fn first(&mut self) -> Budget;

    /// The budget of the next pass, given what the last one did; `None` when
    /// there is nothing larger to try, which ends the search.
    fn next(&mut self, last: &PassReport) -> Option<Budget>;
}

/// The schedules available, as a parameter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScheduleKind {
    /// `(discrepancy, split)` pairs in order of increasing sum:
    /// `(0,0)`, `(0,1) (1,0)`, `(0,2) (1,1) (2,0)`, …
    #[default]
    Diagonal,
    /// Grows a square: every budget with `max(d, s) = k` before any with
    /// `k + 1`, each shell ending on `(k, k)`, which contains every budget
    /// before it.
    Square,
}

impl ScheduleKind {
    /// Every schedule, in declaration order.
    pub const ALL: [Self; 2] = [Self::Diagonal, Self::Square];

    /// The spelling used on the command line and in the Python API.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Diagonal => "diagonal",
            Self::Square => "square",
        }
    }

    /// Creates the schedule for a problem with the given bounds.
    pub fn build(self, bounds: ScheduleBounds) -> Box<dyn BudgetSchedule> {
        match self {
            Self::Diagonal => Box::new(Diagonal::new(bounds)),
            Self::Square => Box::new(Square::new(bounds)),
        }
    }
}

impl fmt::Display for ScheduleKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ScheduleKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let wanted = s.trim().to_ascii_lowercase();
        Self::ALL
            .into_iter()
            .find(|kind| kind.as_str() == wanted)
            .ok_or_else(|| {
                let names: Vec<_> = Self::ALL.iter().map(|k| k.as_str()).collect();
                format!(
                    "unknown budget schedule `{wanted}` (expected one of: {})",
                    names.join(", ")
                )
            })
    }
}

/// `(discrepancy, split)` pairs in order of increasing sum.
///
/// Walks the diagonals `(0,0)`, `(0,1) (1,0)`, `(0,2) (1,1) (2,0)`, … and,
/// within one, from the smallest discrepancy up. Pairs outside
/// [`ScheduleBounds`] are skipped; the last budget is the unrestricted search.
/// Each budget is handed out once.
struct Diagonal {
    max_discrepancy: usize,
    /// `None` when no split is allowed at all: the schedule is then empty.
    max_split_discrepancy: Option<usize>,
    /// The last budget handed out.
    current: Budget,
}

impl Diagonal {
    fn new(bounds: ScheduleBounds) -> Self {
        Self {
            max_discrepancy: bounds.max_discrepancy,
            max_split_discrepancy: bounds.max_splits.checked_sub(1),
            current: Budget {
                discrepancy: 0,
                split_discrepancy: 0,
            },
        }
    }
}

impl BudgetSchedule for Diagonal {
    fn first(&mut self) -> Budget {
        self.current
    }

    fn next(&mut self, _last: &PassReport) -> Option<Budget> {
        let max_s = self.max_split_discrepancy?;
        let max_d = self.max_discrepancy;
        let mut sum = self.current.discrepancy + self.current.split_discrepancy;
        let mut d = self.current.discrepancy + 1;
        while sum <= max_d + max_s {
            // On diagonal `sum`, `d` ranges over `sum - max_s ..= min(sum, max_d)`.
            d = d.max(sum.saturating_sub(max_s));
            if d <= sum.min(max_d) {
                self.current = Budget {
                    discrepancy: d,
                    split_discrepancy: sum - d,
                };
                return Some(self.current);
            }
            sum += 1;
            d = 0;
        }
        None
    }
}

/// Budgets in growing squares.
///
/// Shell `k` holds every `(d, s)` with `max(d, s) = k`, visited as
/// `(k, 0) .. (k, k-1)`, then `(0, k) .. (k-1, k)`, then `(k, k)`:
///
/// ```text
/// (0,0) | (1,0) (0,1) (1,1) | (2,0) (2,1) (0,2) (1,2) (2,2) | ...
/// ```
///
/// Within a shell the budget can still shrink in one dimension (`(1,0)` to
/// `(0,1)`), but every shell ends on the budget that contains all the ones
/// before it, which [`Diagonal`] never does. Pairs outside [`ScheduleBounds`] are
/// skipped; the last budget is the unrestricted search. When the split budget
/// does not apply, only the discrepancy axis is walked, since passes differing
/// only in `s` would be identical.
struct Square {
    max_discrepancy: usize,
    max_split_discrepancy: usize,
    shell: usize,
    pending: std::collections::VecDeque<Budget>,
}

impl Square {
    fn new(bounds: ScheduleBounds) -> Self {
        Self {
            max_discrepancy: bounds.max_discrepancy,
            max_split_discrepancy: if bounds.split_applies {
                bounds.max_splits.saturating_sub(1)
            } else {
                0
            },
            shell: 0,
            pending: Default::default(),
        }
    }

    /// The budgets of shell `k` that lie within bounds, in visiting order.
    fn fill(&mut self, k: usize) {
        let (max_d, max_s) = (self.max_discrepancy, self.max_split_discrepancy);
        let pairs = (0..k)
            .map(|s| (k, s))
            .chain((0..k).map(|d| (d, k)))
            .chain(std::iter::once((k, k)));
        self.pending
            .extend(pairs.filter(|&(d, s)| d <= max_d && s <= max_s).map(
                |(discrepancy, split_discrepancy)| Budget {
                    discrepancy,
                    split_discrepancy,
                },
            ));
    }
}

impl BudgetSchedule for Square {
    fn first(&mut self) -> Budget {
        Budget {
            discrepancy: 0,
            split_discrepancy: 0,
        }
    }

    fn next(&mut self, _last: &PassReport) -> Option<Budget> {
        // Shell 0 is the first budget; start from shell 1.
        while self.pending.is_empty() {
            self.shell += 1;
            if self.shell > self.max_discrepancy.max(self.max_split_discrepancy) {
                return None;
            }
            self.fill(self.shell);
        }
        self.pending.pop_front()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn walk(kind: ScheduleKind, bounds: ScheduleBounds) -> Vec<(usize, usize)> {
        let mut schedule = kind.build(bounds);
        let mut out = vec![schedule.first()];
        while let Some(b) = schedule.next(&PassReport::default()) {
            out.push(b);
        }
        out.iter()
            .map(|b| (b.discrepancy, b.split_discrepancy))
            .collect()
    }

    #[test]
    fn the_diagonal_walks_by_sum_and_never_repeats_a_budget() {
        let bounds = ScheduleBounds {
            max_discrepancy: 2,
            max_splits: 3,
            split_applies: true,
        };
        let seen = walk(ScheduleKind::Diagonal, bounds);
        let expected = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 2),
            (1, 1),
            (2, 0),
            (1, 2),
            (2, 1),
            (2, 2),
        ];
        assert_eq!(seen, expected);
        let mut distinct = seen.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), seen.len());
    }

    #[test]
    fn the_square_grows_shell_by_shell() {
        let bounds = ScheduleBounds {
            max_discrepancy: 2,
            max_splits: 3,
            split_applies: true,
        };
        assert_eq!(
            walk(ScheduleKind::Square, bounds),
            vec![
                (0, 0),
                (1, 0),
                (0, 1),
                (1, 1),
                (2, 0),
                (2, 1),
                (0, 2),
                (1, 2),
                (2, 2),
            ]
        );
    }

    #[test]
    fn the_square_covers_the_rectangle_once_and_ends_unrestricted() {
        let bounds = ScheduleBounds {
            max_discrepancy: 3,
            max_splits: 6,
            split_applies: true,
        };
        let seen = walk(ScheduleKind::Square, bounds);
        let mut sorted = seen.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), seen.len(), "a budget was repeated");
        assert_eq!(seen.len(), 4 * 6, "every budget in the rectangle, once");
        assert_eq!(*seen.last().unwrap(), (3, 5));
    }

    #[test]
    fn each_square_shell_ends_on_a_budget_containing_all_before_it() {
        let bounds = ScheduleBounds {
            max_discrepancy: 4,
            max_splits: 5,
            split_applies: true,
        };
        let seen = walk(ScheduleKind::Square, bounds);
        for (i, &(d, s)) in seen.iter().enumerate() {
            if d == s {
                assert!(seen[..i].iter().all(|&(pd, ps)| pd <= d && ps <= s));
            }
        }
    }

    #[test]
    fn without_a_split_budget_the_square_walks_only_the_discrepancy() {
        let bounds = ScheduleBounds {
            max_discrepancy: 3,
            max_splits: 10,
            split_applies: false,
        };
        assert_eq!(
            walk(ScheduleKind::Square, bounds),
            vec![(0, 0), (1, 0), (2, 0), (3, 0)]
        );
    }

    #[test]
    fn schedules_parse_from_their_own_names() {
        for kind in ScheduleKind::ALL {
            assert_eq!(kind.as_str().parse::<ScheduleKind>(), Ok(kind));
        }
        assert!("sideways".parse::<ScheduleKind>().is_err());
    }
}
