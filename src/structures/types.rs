pub struct BitsetStructData {
    pub(crate) inputs: Vec<Vec<u64>>,
    pub(crate) targets: Vec<Vec<u64>>,
    pub(crate) chunks: usize,
    pub(crate) size: usize,
}

impl Default for BitsetStructData {
    fn default() -> Self {
        Self {
            inputs: vec![],
            targets: vec![],
            chunks: 0,
            size: 0,
        }
    }
}

pub struct DoublePointerData {
    pub(crate) inputs: Vec<Vec<usize>>,
    pub(crate) target: Option<Vec<usize>>,
    pub(crate) num_labels: usize,
    pub(crate) num_attributes: usize,
}
