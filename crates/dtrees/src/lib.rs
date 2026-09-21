// A library hands text back to its caller rather than printing it; the
// binaries and examples print. Tests may print.
#![cfg_attr(not(test), warn(clippy::print_stdout, clippy::print_stderr))]

pub mod algorithms;
pub mod bitsets;
pub mod caching;
pub mod cover;
pub mod example_parser;
pub mod globals;
pub mod parsers;
pub mod reader;
pub mod tree;
