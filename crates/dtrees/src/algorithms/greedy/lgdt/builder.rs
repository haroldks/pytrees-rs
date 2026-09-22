use crate::algorithms::common::config::BaseSearchConfig;
use crate::algorithms::common::errors::NativeError;
use crate::algorithms::greedy::lgdt::LGDT;
use crate::algorithms::optimal::depth2::{ErrorMinimizer, InfoGainMaximizer, OptimalDepth2Tree};
use crate::tree::Tree;

/// Builder for [`LGDT`]. A depth-2 solver is required.
///
/// ```
/// use dtrees_rs::algorithms::greedy::factories::with_error_minimizer;
///
/// let lgdt = with_error_minimizer().max_depth(5).min_support(5).build();
/// assert!(lgdt.is_ok());
/// ```
pub struct LGDTBuilder<D>
where
    D: OptimalDepth2Tree + ?Sized,
{
    config: BaseSearchConfig,
    search: Option<Box<D>>,
}

impl<D> Default for LGDTBuilder<D>
where
    D: OptimalDepth2Tree + ?Sized,
{
    fn default() -> Self {
        Self {
            config: BaseSearchConfig::default(),
            search: None,
        }
    }
}

impl<D> LGDTBuilder<D>
where
    D: OptimalDepth2Tree + ?Sized,
{
    /// A builder whose lookahead maximises information gain.
    pub fn with_default_info_gain_maximizer() -> LGDTBuilder<InfoGainMaximizer<NativeError>> {
        LGDTBuilder::default().search(Box::<InfoGainMaximizer<NativeError>>::default())
    }

    /// A builder whose lookahead minimises the error.
    pub fn with_default_error_minimizer() -> LGDTBuilder<ErrorMinimizer<NativeError>> {
        LGDTBuilder::default().search(Box::<ErrorMinimizer<NativeError>>::default())
    }

    /// Minimum number of instances in each leaf.
    pub fn min_support(mut self, value: usize) -> Self {
        self.config.min_support = value;
        self
    }

    /// Maximum depth of the tree.
    pub fn max_depth(mut self, value: usize) -> Self {
        self.config.max_depth = value;
        self
    }

    /// Only trees with a lower error are accepted.
    pub fn max_error(mut self, value: f64) -> Self {
        self.config.max_error = value;
        self
    }

    /// Time limit in seconds.
    pub fn max_time(mut self, value: f64) -> Self {
        self.config.max_time = value;
        self
    }

    /// The depth-2 solver used for the lookahead.
    pub fn search(mut self, value: Box<D>) -> Self {
        self.search = Some(value);
        self
    }

    /// Builds the learner, or says which required part is missing.
    pub fn build(self) -> Result<LGDT<D>, String> {
        let search = self
            .search
            .ok_or("Optimal Depth two Search algorithm is required")?;
        Ok(LGDT {
            search,
            config: self.config,
            tree: Tree::default(),
        })
    }
}
