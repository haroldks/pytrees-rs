//! The cost of a leaf, and whether threshold skipping may assume it behaves
//! like misclassification.

use crate::common::{classification_error, Cost};

/// What a custom loss receives for a leaf.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LossInput {
    /// The number of instances of each class in the leaf.
    ClassCounts,
    /// The ids of the leaf's instances (their row in the training data).
    Indices,
}

/// A leaf, as a custom loss sees it.
#[derive(Copy, Clone, Debug)]
pub enum Leaf<'a> {
    /// The number of instances of each class.
    ClassCounts(&'a [usize]),
    /// The ids of the instances.
    Indices(&'a [usize]),
}

/// A caller's leaf loss: `(cost, class predicted)`. The cost of a tree is the
/// sum of the costs of its leaves, and a cost must be non-negative.
pub trait LeafLoss: Send + Sync {
    fn cost(&self, leaf: Leaf<'_>) -> (Cost, usize);
}

/// The loss a search minimises.
#[derive(Default)]
pub enum Loss {
    /// Misclassification: the number of instances not of the majority
    /// class.
    #[default]
    Misclassification,
    /// A caller's loss, with the input it asks for.
    Custom {
        function: Box<dyn LeafLoss>,
        input: LossInput,
    },
}

impl Loss {
    /// A caller's loss.
    pub fn custom(function: Box<dyn LeafLoss>, input: LossInput) -> Self {
        Loss::Custom { function, input }
    }

    /// Whether this is the built-in misclassification.
    pub fn is_misclassification(&self) -> bool {
        matches!(self, Loss::Misclassification)
    }

    /// Whether the loss reads instance ids rather than class counts.
    pub fn uses_indices(&self) -> bool {
        matches!(
            self,
            Loss::Custom {
                input: LossInput::Indices,
                ..
            }
        )
    }

    /// The cost of a leaf with these class counts and instance ids.
    /// `indices` is only called for a loss that reads them.
    pub fn leaf(
        &self,
        class_counts: &[usize],
        indices: impl FnOnce() -> Vec<usize>,
    ) -> (Cost, usize) {
        match self {
            Loss::Misclassification => classification_error(class_counts),
            Loss::Custom {
                function,
                input: LossInput::ClassCounts,
            } => function.cost(Leaf::ClassCounts(class_counts)),
            Loss::Custom {
                function,
                input: LossInput::Indices,
            } => function.cost(Leaf::Indices(&indices())),
        }
    }
}

/// Whether the search skips thresholds: the interval pruner, and the skipping
/// and early stops of the depth-2 solver's scan.
///
/// Skipping assumes what misclassification guarantees: moving `n` instances
/// into a leaf never lowers its cost and raises it by at most `n`, and
/// removing them never raises it and lowers it by at most `n`. A custom loss
/// may not, so `Auto` turns skipping off for one. `On` forces it, and the
/// depth-2 solver then counts the moves that break the assumption.
///
/// With a minimum support above 1 the interval pruner stays off whatever the
/// setting, since its rules do not hold there for any loss; `On` then forces
/// only the depth-2 solver's skipping, whose argument does not involve the
/// support.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum Skipping {
    /// On for misclassification, off for a custom loss.
    #[default]
    Auto,
    /// On whatever the loss.
    On,
    /// Off.
    Off,
}

impl Skipping {
    /// Whether thresholds are skipped for this loss.
    ///
    /// The interval pruner adds its own condition, a minimum support of 1;
    /// see `IntervalsPruner::is_sound`.
    pub fn applies(self, loss: &Loss) -> bool {
        match self {
            Skipping::Auto => loss.is_misclassification(),
            Skipping::On => true,
            Skipping::Off => false,
        }
    }
}
