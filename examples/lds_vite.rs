use dtrees_rs::cache::{Caching, Trie};
use dtrees_rs::data::{BinaryData, FileReader};
use dtrees_rs::example::save_results;
use dtrees_rs::heuristics::InformationGain;
use dtrees_rs::searches::errors::NativeError;
use dtrees_rs::searches::optimal::{MonotonicDiscrepancy, LDSDL85};
use dtrees_rs::searches::{
    BranchingStrategy, CacheInitStrategy, LowerBoundStrategy, NodeExposedData, Specialization,
};
use dtrees_rs::structures::RevBitset;

fn main() {
    let data = BinaryData::read("test_data/anneal.txt", false, 0.0);
    let mut structure = RevBitset::new(&data);

    let mut learner = LDSDL85::new(
        1,
        3,
        usize::MAX,
        <f64>::INFINITY,
        300,
        false,
        0,
        CacheInitStrategy::None_,
        Specialization::Murtree,
        LowerBoundStrategy::None_,
        BranchingStrategy::None_,
        NodeExposedData::ClassesSupport,
        Box::<Trie>::default(),
        Box::new(MonotonicDiscrepancy::new(10)),
        Box::<NativeError>::default(),
        Box::<InformationGain>::default(),
    );

    while !learner.time_is_up() {
        let r = learner.partial_fit(&mut structure, None);

        if learner.is_optimal() {
            break;
        }
    }

    println!("{:#?}", learner.statistics);
    learner.tree.print()
}
