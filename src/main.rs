use crate::cache::trie::Trie;
use crate::data::{BinaryData, FileReader};
use crate::heuristics::{GiniIndex, Heuristic, InformationGain, InformationGainRatio, NoHeuristic};
use crate::searches::errors::NativeError;
use crate::searches::optimal::DL85;
use crate::searches::{
    BranchingStrategy, CacheInitStrategy, LowerBoundStrategy, NodeExposedData, Specialization,
};
use crate::structures::RevBitset;
use clap::{arg, Parser};
use std::path::PathBuf;
use std::process;
mod data;
mod globals;
mod heuristics;
mod searches;
mod structures;

mod cache;
mod tree;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Test File path
    #[arg(short, long)]
    file: PathBuf,

    /// Maximum depth
    #[arg(short, long)]
    depth: usize,

    /// Minimum support
    #[arg(short, long, default_value_t = 1)]
    support: usize,

    /// Use Murtree Spacialization Algorithm
    #[arg(short, long)]
    use_specialization: bool,

    /// Lower bound heuristic
    /// 0: None
    /// 1: Similarity
    #[arg(short, long, default_value_t = 0)]
    lower_bound_heuristic: usize,

    /// Branching type
    /// 0: None
    /// 1: Dynamic
    #[arg(short, long, default_value_t = 0)]
    branching_type: usize,

    #[arg(short, long, default_value_t = 0)]
    /// Caching Type
    /// 0 : Trie,
    /// 1: Hashmap (not yet implemented)
    cache_type: usize,

    /// Cache init size
    /// Represents the reserved starting size of the cache
    #[arg(short, long, default_value_t = 0)]
    cache_init_size: usize,

    /// Sorting heuristic
    /// 0: None
    /// 1: Gini
    /// 2: Information Gain
    /// 3: Information Gain Ratio
    #[arg(long, default_value_t = 0)]
    sorting_heuristic: usize,

    /// Data Structure
    /// 0 : Reversible Sparse Bitset
    /// 1 : Classic Bitset
    /// 2 : Double Pointer
    /// 3 : Horizontal Representation
    /// 4 : Raw representation
    #[arg(long, default_value_t = 0)]
    data_structure: usize,

    /// Time limit
    #[arg(long, default_value_t = 600)]
    time_limit: usize,

    /// Max error
    #[arg(long, default_value_t = <f64>::INFINITY)]
    max_error: f64,
}

fn main() {
    let args = Args::parse();
    if !args.file.exists() {
        panic!("File does not exist");
    }

    let file = args.file.to_str().unwrap();

    let specialization = match args.use_specialization {
        true => Specialization::Murtree,
        false => Specialization::None_,
    };

    let lower_bound = match args.lower_bound_heuristic {
        0 => LowerBoundStrategy::None_,
        1 => LowerBoundStrategy::Similarity,
        _ => {
            println!("Invalid lower bound heuristic");
            process::exit(1);
        }
    };

    // TODO : Allow faster use of multiple cache
    // let cache : Box<dyn Caching> =  match args.cache_type {
    //     0 => {Box::<Trie>::default()},
    //     _ =>  {panic!("Invalid number or not yet implemented")}
    // };

    let branching = match args.branching_type {
        0 => BranchingStrategy::None_,
        1 => BranchingStrategy::Dynamic,
        _ => {
            println!("Invalid branching type");
            process::exit(1);
        }
    };

    let heuristic: Box<dyn Heuristic> = match args.sorting_heuristic {
        0 => Box::<NoHeuristic>::default(),
        1 => Box::<GiniIndex>::default(),
        2 => Box::<InformationGain>::default(),
        3 => Box::<InformationGainRatio>::default(),
        _ => {
            println!("Invalid heuristic type");
            process::exit(1);
        }
    };

    let data = BinaryData::read(file, false, 0.0);

    // TODO : Allow support of multiple data structures
    // let mut structure: Box<dyn Structure> = match args.data_structure {
    //     0 => Box::new(RevBitset::new(&data)),
    //     1 => Box::new(Bitset::new(&data)),
    //     2 => Box::new(DoublePointer::new(&data)),
    //     3 => Box::new(Horizontal::new(&data)),
    //     4 => Box::new(RawBinary::new(&data)),
    //     _ => panic!("Invalid data structure")
    // };

    let mut structure = RevBitset::new(&data);

    let error_function = Box::<NativeError>::default();

    // TODO : Add multiple strategies
    let cache = match args.cache_type {
        _ => Box::<Trie>::default(),
    };

    let cache_strategy = match args.cache_init_size == 0 {
        true => CacheInitStrategy::None_,
        false => CacheInitStrategy::UserAllocation,
    };

    let mut learner = DL85::new(
        args.support,
        args.depth,
        args.max_error,
        args.time_limit,
        false,
        args.cache_init_size,
        cache_strategy,
        specialization,
        lower_bound,
        branching,
        NodeExposedData::ClassesSupport,
        cache,
        error_function,
        heuristic,
    );
    learner.fit(&mut structure);
    println!("{:#?}", learner.statistics)
}
