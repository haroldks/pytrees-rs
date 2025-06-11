use crate::algorithms::common::config::BaseSearchConfig;
use crate::algorithms::common::errors::NativeError;
use crate::algorithms::greedy::lgdt::LGDT;
use crate::algorithms::optimal::depth2::{ErrorMinimizer, InfoGainMaximizer, OptimalDepth2Tree};
use crate::tree::Tree;

pub struct LGDTBuilder<D>
where
    D: OptimalDepth2Tree + ?Sized
{
    config: BaseSearchConfig,
    search: Option<Box<D>>
}


impl<D> Default for LGDTBuilder<D>
where 
    D: OptimalDepth2Tree
{
    fn default() -> Self {
        Self {
            config: BaseSearchConfig::default(),
            search: None,
        }
    }
}

impl <D>LGDTBuilder<D>
where 
    D: OptimalDepth2Tree
{
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_default_info_gain_maximizer() -> LGDTBuilder<InfoGainMaximizer<NativeError>> {
        LGDTBuilder::default().search(Box::new(InfoGainMaximizer::default()))
    }

    pub fn with_default_error_minimizer() -> LGDTBuilder<ErrorMinimizer<NativeError>> {
        LGDTBuilder::default().search(Box::new(ErrorMinimizer::default()))
    }
    
    pub fn min_support(mut self, value: usize) -> Self {
        self.config.min_support = value;
        self
    }

    pub fn max_depth(mut self, value: usize) -> Self {
        self.config.max_depth = value;
        self
    }

    pub fn max_error(mut self, value: f64) -> Self {
        self.config.max_error = value;
        self
    }

    pub fn max_time(mut self, value: f64) -> Self {
        self.config.max_time = value;
        self
    }

    pub fn search(mut self, value: Box<D>) -> Self {
        self.search = Some(value);
        self
    }

    pub fn build(self) -> Result<LGDT<D>, String> {
        let search = self.search.ok_or("Optimal Depth two Search algorithm is required")?;
        Ok(LGDT {
            search,
            config: self.config,
            tree: Tree::default()
        })
    }
    
}


pub mod default_builders {
    use super::*;

   
   
}
