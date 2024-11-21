use dtrees_rs::cache::Trie;
use dtrees_rs::data::{BinaryData, FileReader};
use dtrees_rs::heuristics::{InformationGain, NoHeuristic};
use dtrees_rs::searches::errors::NativeError;
use dtrees_rs::searches::optimal::{
    ExponentialDiscrepancy, LubyDiscrepancy, MonotonicDiscrepancy, DL85, LDSDL85,
};
use dtrees_rs::searches::{
    BranchingStrategy, CacheInitStrategy, LowerBoundStrategy, NodeExposedData, Specialization,
};
use dtrees_rs::structures::RevBitset;

fn main() {
    let data = BinaryData::read("test_data/anneal.txt", false, 0.0);
    let mut structure = RevBitset::new(&data);
    let error_function = Box::<NativeError>::default();
    let discrepancy_strat = Box::<ExponentialDiscrepancy>::default();
    let cache = Box::<Trie>::default();
    let heuristics = Box::<NoHeuristic>::default();

    let mut learner = LDSDL85::new(
        1,
        4,
        <usize>::MAX,
        <f64>::INFINITY,
        600,
        true,
        0,
        CacheInitStrategy::None_,
        Specialization::None_,
        LowerBoundStrategy::None_,
        BranchingStrategy::None_,
        NodeExposedData::ClassesSupport,
        cache,
        discrepancy_strat,
        error_function,
        heuristics,
    );
    learner.fit(&mut structure);
    println!("{:#?}", learner.statistics)
}
