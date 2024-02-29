use crate::data::FileReader;
use crate::structures::types::BitsetStructData;

// Structure to export from the module
pub use bs::Bitset;
pub use dp::DoublePointer;
pub use hs::Horizontal;
pub use rsbs::RevBitset;
pub use rws::RawBinary;

// In out difference between data
pub type Difference = (usize, usize);
mod bs;
mod dp;
mod hs;
mod rsbs;
mod rws;
mod types;

pub trait Structure {
    fn num_attributes(&self) -> usize;
    fn num_labels(&self) -> usize;
    fn label_support(&self, label: usize) -> usize;
    fn labels_support(&mut self) -> &[usize];
    fn support(&mut self) -> usize;
    fn get_support(&self) -> usize;
    fn push(&mut self, item: usize) -> usize;
    fn backtrack(&mut self);
    fn temp_push(&mut self, item: usize) -> usize;
    fn reset(&mut self);
    fn get_position(&self) -> &[usize];
    fn change_position(&mut self, itemset: &[usize]) -> usize {
        self.reset();
        for item in itemset {
            self.push(*item);
        }
        self.support()
    }

    fn get_data_cover(&mut self) -> DataCover;

    fn get_difference(&self, data_cover: &DataCover) -> Difference;

    fn get_tids(&self) -> Vec<usize>;
}

pub fn format_data_into_bitset<T>(data: &T) -> BitsetStructData
where
    T: FileReader,
{
    let data_ref = data.get_train();
    let num_labels = data.num_labels();
    let size = data.train_size();
    let num_attributes = data.num_attributes();

    let mut chunks = 1usize;
    if size > 64 {
        chunks = match size % 64 {
            0 => size / 64,
            _ => (size / 64) + 1,
        };
    }

    let mut inputs = vec![vec![0u64; chunks]; num_attributes];
    let mut targets = match num_labels == 0 {
        true => {
            vec![]
        }
        false => {
            vec![vec![0u64; chunks]; num_labels]
        }
    };

    for (tid, row) in data_ref.1.iter().rev().enumerate() {
        let row_chunk = chunks - 1 - tid / 64;
        for (i, val) in row.iter().enumerate() {
            if *val == 1 {
                inputs[i][row_chunk] |= 1u64 << (tid % 64);
            }
        }
        if data_ref.0.is_some() {
            let class = data_ref
                .0
                .as_ref()
                .map_or(0, |target| target[size - 1 - tid]);
            targets[class][row_chunk] |= 1u64 << (tid % 64);
        }
    }

    BitsetStructData {
        inputs,
        targets,
        chunks,
        size,
    }
}

#[derive(Clone)]
pub struct DataCover {
    cover: Vec<u64>, // u64 because of the bitset
    limit: isize,
    index: Vec<usize>,
    pub(crate) error: f64,
    support: usize,
}

impl DataCover {
    pub fn new() -> Self {
        Self {
            cover: vec![],
            limit: 0,
            index: vec![],
            error: 0.0,
            support: 0,
        }
    }

    pub fn update(&mut self, data_cover: DataCover) {
        *self = data_cover
    }

    pub fn difference(&self, data_cover: DataCover) {}
}

impl Default for DataCover {
    fn default() -> Self {
        Self {
            cover: vec![],
            limit: 0,
            index: vec![],
            error: 0.0,
            support: 0,
        }
    }
}
