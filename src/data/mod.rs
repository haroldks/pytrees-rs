pub mod binary_data;

pub use binary_data::BinaryData;
use ndarray::{Array, IxDyn};
use std::fs::File;
use std::io::{BufRead, BufReader, Error};

pub type Data = (Option<Vec<usize>>, Vec<Vec<usize>>);

pub trait FileReader {
    fn read(filename: &str, shuffle: bool, split: f64) -> Self;

    fn read_from_numpy(input: &Array<usize, IxDyn>, target: Option<&Array<usize, IxDyn>>) -> Self;

    fn size(&self) -> usize;

    fn num_labels(&self) -> usize;

    fn num_attributes(&self) -> usize;

    fn get_train(&self) -> &Data;

    fn train_size(&self) -> usize;

    fn open_file(filename: &str) -> Result<Vec<String>, Error> {
        let input = File::open(filename)?; //Error Handling for missing filename
        let buffered = BufReader::new(input); // Buffer for the file
        Ok(buffered
            .lines()
            .map(|x| x.unwrap())
            .collect::<Vec<String>>())
    }
}
