use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SearchStrategy {
    Depth2ErrorMinimizer,
    Depth2InfoGainMaximize
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FitError {
    InvalidDepth(usize),
    InvalidMinSupport(usize),
    EmptyCandidates,
    AlgorithmError,
    InsufficientData,
    LGDTEmptyTree
}