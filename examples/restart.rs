use dtrees_rs::cache::Trie;
use dtrees_rs::data::BinaryData;
use dtrees_rs::data::FileReader;
use dtrees_rs::heuristics::{InformationGain, NoHeuristic};
use dtrees_rs::searches::errors::NativeError;
use dtrees_rs::searches::optimal::RestartDL85;
use dtrees_rs::searches::{
    BranchingStrategy, CacheInitStrategy, LowerBoundStrategy, NodeExposedData, Specialization,
};
use dtrees_rs::structures::RevBitset;

fn main() {
    let data = BinaryData::read("test_data/kr-vs-kp.txt", false, 0.0);
    let mut structure = RevBitset::new(&data);
    let error_function = Box::<NativeError>::default();
    let cache = Box::<Trie>::default();
    let heuristics = Box::<NoHeuristic>::default();

    let mut learner = RestartDL85::new(
        1,
        4,
        <f64>::INFINITY,
        None,
        600,
        true,
        0,
        CacheInitStrategy::None_,
        Specialization::None_,
        LowerBoundStrategy::None_,
        BranchingStrategy::None_,
        NodeExposedData::ClassesSupport,
        cache,
        error_function,
        heuristics,
    );

    learner.fit(&mut structure);
    println!("{:?}", learner.statistics);
}
