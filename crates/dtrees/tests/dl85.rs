use float_cmp::{ApproxEq, F64Margin};
use paste::paste;
use std::path::Path;

use dtrees_rs::algorithms::common::errors::NativeError;
use dtrees_rs::algorithms::common::heuristics::NoHeuristic;
use dtrees_rs::algorithms::common::types::OptimalDepth2Policy;
use dtrees_rs::algorithms::optimal::depth2::ErrorMinimizer;
use dtrees_rs::algorithms::optimal::dl85::DL85Builder;
use dtrees_rs::algorithms::TreeSearchAlgorithm;
use dtrees_rs::caching::Trie;
use dtrees_rs::reader::data_reader::DataReader;

fn solve_dataset(
    dataset_name: &str,
    min_sup: usize,
    max_depth: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    let reader = DataReader::default();

    let path = &format!("test_data/{}.txt", dataset_name);
    let path = Path::new(path);
    let mut cover = reader.read_file(path)?;

    let error_fn = Box::<NativeError>::default();

    let depth2_search = Box::new(ErrorMinimizer::new(error_fn.clone()));

    let mut algo = DL85Builder::default()
        .max_depth(max_depth)
        .min_support(min_sup)
        .max_time(100.0)
        .specialization(OptimalDepth2Policy::Enabled)
        .cache(Box::<Trie>::default())
        .heuristic(Box::<NoHeuristic>::default())
        .depth2_search(depth2_search)
        .error_function(error_fn)
        .build()?;

    algo.fit(&mut cover)?;

    Ok(algo.statistics().tree_error)
}

macro_rules! dl85_test_suite {
    ($name_prefix:ident, $($name:ident: $minsup:expr, $maxdepth:expr, $expected:expr;)*) => {
        $(
            paste! {
                #[test]
                fn [<$name_prefix _ $name _minsup_ $minsup _maxdepth_ $maxdepth>]() -> Result<(), Box<dyn std::error::Error>> {
                    let error = solve_dataset(stringify!($name), $minsup, $maxdepth)?;
                    assert_eq!(
                        error.approx_eq($expected, F64Margin { ulps: 2, epsilon: 0.0 }),
                        true,
                        "Dataset {}: Expected error {}, got {}",
                        stringify!($name),
                        $expected,
                        error
                    );
                    Ok(())
                }
            }
        )*
    }
}

macro_rules! dl85_ignored_test_suite {
    ($name_prefix:ident, $($name:ident: $minsup:expr, $maxdepth:expr, $expected:expr;)*) => {
        $(
            paste! {
                #[test]
                #[ignore]
                fn [<$name_prefix _ $name _minsup_ $minsup _maxdepth_ $maxdepth>]() -> Result<(), Box<dyn std::error::Error>> {
                    let error = solve_dataset(stringify!($name), $minsup, $maxdepth)?;
                    assert_eq!(
                        error.approx_eq($expected, F64Margin { ulps: 2, epsilon: 0.0 }),
                        true,
                        "Dataset {}: Expected error {}, got {}",
                        stringify!($name),
                        $expected,
                        error
                    );
                    Ok(())
                }
            }
        )*
    }
}

dl85_test_suite!(test_d2,
    vehicle: 1, 2, 75.0;
    audiology: 1, 2, 10.0;
    ionosphere: 1, 2, 32.0;
    // segment: 1, 2, 9.0; // Commented out to reduce test runtime
    rsparse_dataset: 1, 2, 0.0;
    anneal: 1, 2, 137.0;
    vote: 1, 2, 17.0;
    ttt: 1, 2, 282.0;
    // pendigits: 1, 2, 153.0; // Commented out to reduce test runtime
    diabetes: 1, 2, 177.0;
    iris: 1, 2, 9.0;
    mushroom: 1, 2, 252.0;
    soybean: 1, 2, 55.0;
    yeast: 1, 2, 437.0;
    hepatitis: 1, 2, 16.0;
    small: 1, 2, 0.0;
    hypothyroid: 1, 2, 70.0;
    iris_multi: 1, 2, 6.0;
    // letter: 1, 2, 599.0; // Commented out to reduce test runtime
    lymph: 1, 2, 22.0;
);

dl85_test_suite!(test_d2,
    small_: 50, 2, 5.0;
    vehicle: 50, 2, 75.0;
    audiology: 50, 2, 11.0;
    ionosphere: 50, 2, 32.0;
    // segment: 50, 2, 21.0;
    rsparse_dataset: 50, 2, 0.0;
    anneal: 50, 2, 164.0;
    vote: 50, 2, 19.0;
    ttt: 50, 2, 282.0;
    // pendigits: 50, 2, 153.0;
    diabetes: 50, 2, 180.0;
    mushroom: 50, 2, 252.0;
    soybean: 50, 2, 85.0;
    yeast: 50, 2, 441.0;
    small: 50, 2, 2.0;
    hypothyroid: 50, 2, 70.0;
    // letter: 50, 2, 599.0;
);

dl85_ignored_test_suite!(test_d3,
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
