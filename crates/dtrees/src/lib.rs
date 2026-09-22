//! Decision trees over binary features.
//!
//! * [DL8.5](algorithms::optimal::dl85::DL85) learns optimal trees by dynamic
//!   programming with branch-and-bound and caching (Aglin, Nijssen and
//!   Schaus, AAAI 2020). With search rules it becomes anytime: limited
//!   discrepancy search (LDS-DL8.5, ECML PKDD 2022), Top-k, and the general
//!   CA-DL8.5 framework (IDA 2026). See [`algorithms::optimal::rules`].
//! * [LGDT](algorithms::greedy::LGDT) grows a tree greedily, choosing each
//!   test with a depth-2 lookahead (IDA 2024).
//!
//! Data comes as a [`Cover`](cover::Cover), read from a text file by
//! [`DataReader`](reader::data_reader::DataReader): one instance per line,
//! the label first, then the 0/1 features.
//!
//! ```no_run
//! use dtrees_rs::algorithms::greedy::factories::with_error_minimizer;
//! use dtrees_rs::algorithms::TreeSearchAlgorithm;
//! use dtrees_rs::reader::data_reader::DataReader;
//! use std::path::Path;
//!
//! let mut cover = DataReader::default().read_file(Path::new("data.txt"))?;
//! let mut lgdt = with_error_minimizer().max_depth(4).min_support(5).build()?;
//! lgdt.fit(&mut cover)?;
//! println!("{}", lgdt.tree());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// Library code returns text to its caller instead of printing it.
#![cfg_attr(not(test), warn(clippy::print_stdout, clippy::print_stderr))]

pub mod algorithms;
pub mod bitsets;
pub mod caching;
pub mod cover;
pub mod globals;
#[cfg(feature = "cli")]
pub mod parsers;
pub mod reader;
pub mod tree;
