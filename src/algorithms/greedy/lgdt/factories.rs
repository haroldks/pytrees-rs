use crate::algorithms::common::errors::NativeError;
use crate::algorithms::greedy::lgdt::builder::LGDTBuilder;
use crate::algorithms::optimal::depth2::{ErrorMinimizer, InfoGainMaximizer};

pub fn with_info_gain() -> LGDTBuilder<InfoGainMaximizer<NativeError>> {
    LGDTBuilder::default()
        .search(Box::new(InfoGainMaximizer::default()))
}

pub fn with_error_minimizer() -> LGDTBuilder<ErrorMinimizer<NativeError>> {
    LGDTBuilder::default()
        .search(Box::new(ErrorMinimizer::default()))
}