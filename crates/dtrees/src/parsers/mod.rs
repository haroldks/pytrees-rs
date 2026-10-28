//! Command line parsing for the examples and the `dtrees` CLI.

use crate::caching::{Caching, CoverMap, Trie};
use clap::ValueEnum;

pub mod examples;

/// The cache of the subproblems.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum CacheType {
    /// A trie keyed by the path of tests ([`Trie`]).
    Trie,
    /// A hash map keyed by the depth and the instances ([`CoverMap`]).
    Cover,
}

impl CacheType {
    /// A new, empty cache of this type.
    pub fn new_cache(self) -> Box<dyn Caching> {
        match self {
            CacheType::Trie => Box::<Trie>::default(),
            CacheType::Cover => Box::<CoverMap>::default(),
        }
    }
}
