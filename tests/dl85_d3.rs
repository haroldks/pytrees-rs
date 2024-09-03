use dtrees_rs::cache::Trie;
use dtrees_rs::data::{BinaryData, FileReader};
use dtrees_rs::heuristics::NoHeuristic;
use dtrees_rs::searches::errors::NativeError;
use dtrees_rs::searches::optimal::DL85;
use dtrees_rs::searches::{
    BranchingStrategy, CacheInitStrategy, LowerBoundStrategy, NodeExposedData, Specialization,
};
use dtrees_rs::structures::RevBitset;
use float_cmp::{ApproxEq, F64Margin};
use paste::paste;

macro_rules! tests_dl85_d3 {
    ($($name:ident: $minsup:expr, $maxdepth:expr, $value:expr;)*) => {
        $(
            paste!{
                 #[test]
                 #[ignore]
                 fn [<blossom_ $name _minsup_ $minsup _maxdepth_ $maxdepth >]() {
                    let dataset = BinaryData::read(&format!("test_data/{}.txt", stringify!($name)), false, 0.0);
                    let mut structure = RevBitset::new(&dataset);
                    let error = solve_instance(&mut structure, $minsup, $maxdepth);
                    assert_eq!(error.approx_eq($value, F64Margin { ulps: 2, epsilon: 0.0 }), true);
                }

            }
        )*
    }
}

fn solve_instance(structure: &mut RevBitset, min_sup: usize, max_depth: usize) -> f64 {
    let mut search = DL85::new(
        min_sup,
        max_depth,
        <f64>::INFINITY,
        300,
        true,
        0,
        CacheInitStrategy::None_,
        Specialization::None_,
        LowerBoundStrategy::None_,
        BranchingStrategy::None_,
        NodeExposedData::ClassesSupport,
        Box::new(Trie::default()),
        Box::new(NativeError::default()),
        Box::new(NoHeuristic::default()),
    );
    search.fit(structure);
    search.statistics.tree_error
}

tests_dl85_d3!(
    vehicle: 1, 3, 26.0;
    audiology: 1, 3, 5.0;
    ionosphere: 1, 3, 22.0;
    segment: 1, 3, 0.0;
    rsparse_dataset: 1, 3, 0.0;
    anneal: 1, 3, 112.0;
    vote: 1, 3, 12.0;
    ttt: 1, 3, 216.0;
    pendigits: 1, 3, 47.0;
    diabetes: 1, 3, 162.0;
    iris: 1, 3, 9.0;
    mushroom: 1, 3, 8.0;
    soybean: 1, 3, 29.0;
    yeast: 1, 3, 403.0;
    hepatitis: 1, 3, 10.0;
    small: 1, 3, 0.0;
    hypothyroid: 1, 3, 61.0;
    iris_multi: 1, 3, 2.0;
    letter: 1, 3, 369.0;
    lymph: 1, 3, 12.0;
);
