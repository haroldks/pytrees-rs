use dtrees_rs::algorithms::common::errors::NativeError;
use dtrees_rs::algorithms::common::heuristics::InformationGain;
use dtrees_rs::algorithms::common::types::OptimalDepth2Policy;
use dtrees_rs::algorithms::optimal::depth2::ErrorMinimizer;
use dtrees_rs::algorithms::optimal::dl85::HashDL85Builder;

use dtrees_rs::algorithms::TreeSearchAlgorithm;

use dtrees_rs::reader::data_reader::DataReader;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reader = DataReader::default();
    let path = Path::new("test_data/anneal.txt");
    let mut cover = reader.read_file(path)?;

    let error_fn = Box::<NativeError>::default();

    let depth2 = Box::new(ErrorMinimizer::new(error_fn.clone()));

    let mut algo = HashDL85Builder::default()
        .max_depth(3)
        .min_support(1)
        .max_time(300.0)
        .always_sort(true)
        .lookahead_depth(1, Some(2), 0)
        .specialization(OptimalDepth2Policy::Enabled)
        .heuristic(Box::<InformationGain>::default())
        .depth2_search(depth2)
        .error_function(error_fn)
        .build()?;

    algo.fit(&mut cover)?;

    println!("Search statistics: {:#?}", algo.statistics());

    algo.tree().print();

    Ok(())
}
