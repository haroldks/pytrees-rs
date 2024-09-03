use dtrees_rs::cache::Trie;
use dtrees_rs::data::{BinaryData, FileReader};
use dtrees_rs::heuristics::InformationGain;
use dtrees_rs::searches::errors::NativeError;
use dtrees_rs::searches::optimal::{LubyDiscrepancy, LDSDL85};
use dtrees_rs::searches::{
    BranchingStrategy, CacheInitStrategy, LowerBoundStrategy, NodeExposedData, Specialization,
};
use dtrees_rs::structures::RevBitset;
use float_cmp::{ApproxEq, F64Margin};
use paste::paste;

macro_rules! tests_ldsdl85_d2 {
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
    let mut search = LDSDL85::new(
        min_sup,
        max_depth,
        <usize>::MAX,
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
        Box::new(LubyDiscrepancy::default()),
        Box::new(NativeError::default()),
        Box::new(InformationGain::default()),
    );
    search.fit(structure);
    search.statistics.tree_error
}

tests_ldsdl85_d2!(
    vehicle: 1, 2, 75.0;
    audiology: 1, 2, 10.0;
    ionosphere: 1, 2, 32.0;
    // segment: 1, 2, 9;
    rsparse_dataset: 1, 2, 0.0;
    anneal: 1, 2, 137.0;
    vote: 1, 2, 17.0;
    ttt: 1, 2, 282.0;
    // pendigits: 1, 2, 153;
    diabetes: 1, 2, 177.0;
    iris: 1, 2, 9.0;
    mushroom: 1, 2, 252.0;
    soybean: 1, 2, 55.0;
    yeast: 1, 2, 437.0;
    hepatitis: 1, 2, 16.0;
    small: 1, 2, 0.0;
    hypothyroid: 1, 2, 70.0;
    iris_multi: 1, 2, 6.0;
    // letter: 1, 2, 599;
    lymph: 1, 2, 22.0;
);

tests_ldsdl85_d2!(

    small_: 50, 2, 5.0;
    vehicle: 50, 2, 75.0;
    audiology: 50, 2, 11.0;
    ionosphere: 50, 2, 32.0;
    // segment: 50, 2, 21;
    rsparse_dataset: 50, 2, 0.0;
    anneal: 50, 2, 164.0;
    vote: 50, 2, 19.0;
    ttt: 50, 2, 282.0;
    // pendigits: 50, 2, 153;
    diabetes: 50, 2, 180.0;
    mushroom: 50, 2, 252.0;
    soybean: 50, 2, 85.0;
    yeast: 50, 2, 441.0;
    small: 50, 2, 2.0;
    hypothyroid: 50, 2, 70.0;
    // letter: 50, 2, 599;
);
