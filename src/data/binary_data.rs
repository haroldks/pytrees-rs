use crate::data::{Data, FileReader};
use ndarray::{Array, IxDyn};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashSet;

#[derive(Clone)]
pub struct BinaryData {
    filename: String,
    shuffle: bool,
    split: f64,
    train: Data,
    test: Option<Data>,
    size: usize,
    train_size: usize,
    num_labels: usize,
    num_attributes: usize,
}

impl FileReader for BinaryData {
    fn read(filename: &str, shuffle: bool, split: f64) -> Self {
        let mut data = Self::open_file(filename).unwrap();
        let size = data.len();

        if shuffle {
            data.shuffle(&mut thread_rng())
        }

        let test_size = (size as f64 * split) as usize;

        let test = match test_size >= 1 {
            true => Some(BinaryData::create_set(
                data.drain(0..test_size).collect::<Vec<String>>(),
            )),
            false => None,
        };

        let train = BinaryData::create_set(data);
        let train_size = train.1.len();
        let num_attributes = train.1[0].len();
        let num_labels = train
            .0
            .as_ref()
            .map_or(0, |elem| elem.iter().collect::<HashSet<_>>().len());
        Self {
            filename: filename.to_string(),
            shuffle,
            split,
            train,
            test,
            size,
            train_size,
            num_labels,
            num_attributes,
        }
    }

    fn read_from_numpy(input: &Array<usize, IxDyn>, target: Option<&Array<usize, IxDyn>>) -> Self {
        let targets = match target.is_some() {
            true => Some(target.unwrap().clone().into_raw_vec()),
            false => None,
        };

        let mut inputs = vec![];
        for row in input.rows() {
            inputs.push(row.to_vec());
        }
        let train_size = inputs.len();
        let num_attributes = inputs[0].len();
        let num_labels = targets
            .as_ref()
            .map_or(0, |elem| elem.iter().collect::<HashSet<_>>().len());
        let train: Data = (targets, inputs);

        Self {
            filename: "from_python".to_string(),
            shuffle: false,
            split: 0.0f64,
            train,
            test: None,
            size: train_size,
            train_size,
            num_labels,
            num_attributes,
        }
    }

    fn size(&self) -> usize {
        self.size
    }

    fn num_labels(&self) -> usize {
        self.num_labels
    }

    fn num_attributes(&self) -> usize {
        self.num_attributes
    }

    fn get_train(&self) -> &Data {
        &self.train
    }

    fn train_size(&self) -> usize {
        self.train_size
    }
}

impl BinaryData {
    fn create_set(data: Vec<String>) -> Data {
        let data = data
            .iter()
            .map(|line| {
                line.split_whitespace()
                    .map(|y| y.parse().unwrap())
                    .collect::<Vec<usize>>()
            })
            .collect::<Vec<Vec<usize>>>();
        let targets = data.iter().map(|row| row[0]).collect::<Vec<usize>>();
        let rows = data
            .iter()
            .map(|row| row[1..].to_vec())
            .collect::<Vec<Vec<usize>>>();
        (Some(targets), rows)
    }
}

#[cfg(test)]
mod binary_data_test {
    use crate::data::binary_data::BinaryData;
    use crate::data::FileReader;
    use ndarray::{arr1, arr2};
    use std::panic;

    #[test]
    fn can_open_file() {
        let dataset = BinaryData::open_file("test_data/small.txt");

        let _dataset = match dataset {
            Ok(file) => file,
            Err(_error) => {
                panic!("Should not panic")
            }
        };
    }

    #[test]
    #[should_panic(expected = "Missing File")]
    fn missing_file() {
        let dataset = BinaryData::open_file("test_data/missing.txt");

        let _dataset = match dataset {
            Ok(file) => file,
            Err(_error) => {
                panic!("Missing File")
            }
        };
    }

    #[test]
    fn data_is_retrieved() {
        let dataset = BinaryData::open_file("test_data/small.txt");
        let content = vec!["0 1 0 1", "0 0 1 1", "1 0 0 0", "1 0 1 0"];

        let dataset = match dataset {
            Ok(file) => file,
            Err(_) => {
                panic!("Should not panic")
            }
        };
        assert_eq!(dataset.iter().eq(content.iter()), true);
    }

    #[test]
    fn binary_dataset_no_shuffle_and_no_split() {
        let dataset = BinaryData::read("test_data/small.txt", false, 0.0);
        assert_eq!(dataset.filename, "test_data/small.txt");
        assert_eq!(dataset.shuffle, false);
        assert_eq!(dataset.test.is_none(), true);
    }

    #[test]
    fn binary_dataset_no_shuffle_and_half_split() {
        let mut dataset = BinaryData::read("test_data/small.txt", false, 0.5);
        assert_eq!(dataset.test.is_some(), true);
        let data = dataset.test.take().unwrap();
        let rows = data.1;
        assert_eq!(data.0.map_or(0, |d| d.len()), dataset.size() / 2);
        let content = vec![vec![1, 0, 1], vec![0, 1, 1]];
        assert_eq!(rows.iter().eq(content.iter()), true);
    }

    #[test]
    fn binary_dataset_shuffled_and_half_split() {
        let mut dataset = BinaryData::read("test_data/small.txt", true, 0.25);
        assert_eq!(dataset.test.is_some(), true);
        let data = dataset.test.take().unwrap();
        assert_eq!(data.0.map_or(0, |d| d.len()), 1);
    }

    #[test]
    fn binary_dataset_size_and_label() {
        let dataset = BinaryData::read("test_data/small.txt", true, 0.0);
        assert_eq!(dataset.size(), 4);
        assert_eq!(dataset.num_labels(), 2);
    }

    #[test]
    fn binary_dataset_numpy() {
        let targets = arr1(&[0usize, 0, 1, 1]).into_dyn();
        let input = arr2(&[[1usize, 0, 1], [0, 1, 1], [0, 0, 0], [0, 1, 0]]).into_dyn();
        let dataset = BinaryData::read_from_numpy(&input, Some(&targets));

        assert_eq!(dataset.size(), 4);
        assert_eq!(dataset.num_labels(), 2);
        assert_eq!(dataset.shuffle, false);
        assert_eq!(dataset.test.is_none(), true);
    }
}
