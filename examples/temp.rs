use dtrees_rs::data::{BinaryData, FileReader};

fn main() {
    let data = BinaryData::read("test_data/paper.txt", false, 0.0);
    let train = data.get_train();
    let nb_feats = data.num_attributes();
    let mut array = vec![vec![]; nb_feats];
    for transaction in train.1.iter() {
        for (feat, elem) in transaction.iter().enumerate() {
            array[feat].push(*elem);
        }
    }

    println!("{:?}", array)
}
