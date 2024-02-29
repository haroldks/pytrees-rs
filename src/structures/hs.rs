use crate::data::FileReader;
use crate::globals::{attribute, item_type};
use crate::structures::{DataCover, Difference, Structure};
use std::collections::HashSet;

#[derive(Clone)]
pub struct Horizontal {
    input: Vec<Vec<Vec<usize>>>,
    support: usize,
    labels_support: Vec<usize>,
    num_labels: usize,
    num_attributes: usize,
    position: Vec<usize>,
    state: Vec<Vec<Vec<usize>>>,
}

impl Structure for Horizontal {
    fn num_attributes(&self) -> usize {
        self.num_attributes
    }

    fn num_labels(&self) -> usize {
        self.num_labels
    }

    fn label_support(&self, label: usize) -> usize {
        if self.num_labels == 0 {
            return 0;
        }
        let mut support = <usize>::MAX;
        if label < self.num_labels {
            if let Some(state) = self.get_last_state() {
                support = state[label].len();
            }
        }
        support
    }

    fn labels_support(&mut self) -> &[usize] {
        if !self.labels_support.is_empty() {
            return &self.labels_support;
        }
        self.labels_support.clear();

        if self.num_labels == 0 {
            return &self.labels_support;
        }
        if let Some(state) = self.state.last() {
            for label_state in state.iter() {
                self.labels_support.push(label_state.len());
            }
        }
        &self.labels_support
    }

    fn support(&mut self) -> usize {
        if self.support < <usize>::MAX {
            return self.support;
        }
        if let Some(last) = self.get_last_state() {
            self.support = last.iter().map(|rows| rows.len()).sum();
        }
        self.support
    }

    fn get_support(&self) -> usize {
        self.support
    }

    fn push(&mut self, item: usize) -> usize {
        self.position.push(item);
        self.pushing(item);
        self.support()
    }

    fn backtrack(&mut self) {
        if !self.position.is_empty() {
            self.position.pop();
            self.state.pop();
            self.labels_support.clear();
            self.support = <usize>::MAX;
            // self.support();
        }
    }

    fn temp_push(&mut self, item: usize) -> usize {
        let support = self.push(item);
        self.backtrack();
        support
    }

    fn reset(&mut self) {
        let mut state = Vec::with_capacity(self.num_attributes);
        state.push(self.state[0].clone());
        self.position = Vec::with_capacity(self.num_attributes);
        self.state = state;
        self.support = self.input.iter().map(|label| label.len()).sum::<usize>();
        self.labels_support.clear();
    }
    fn get_position(&self) -> &[usize] {
        &self.position
    }

    fn get_data_cover(&mut self) -> DataCover {
        let mut cover = vec![];
        if let Some(state) = self.state.last() {
            cover = state.iter().flatten().map(|tid| *tid as u64).collect();
        }
        DataCover {
            cover,
            support: self.support(),
            ..DataCover::default()
        }
    }

    fn get_difference(&self, data_cover: &DataCover) -> Difference {
        let mut cover = HashSet::new();
        if let Some(state) = self.state.last() {
            cover = state.iter().flatten().copied().collect::<HashSet<usize>>();
        }
        let saved = data_cover
            .cover
            .iter()
            .map(|val| *val as usize)
            .collect::<HashSet<usize>>();

        let in_count = cover.difference(&saved).collect::<HashSet<_>>().len();
        let out_count = saved.difference(&cover).collect::<HashSet<_>>().len();

        (in_count, out_count)
    }

    fn get_tids(&self) -> Vec<usize> {
        self.state
            .last()
            .unwrap()
            .iter()
            .flatten()
            .cloned()
            .collect()
    }
}

impl Horizontal {
    pub fn format_input_data<T>(data: &T) -> Vec<Vec<Vec<usize>>>
    where
        T: FileReader,
    {
        let data_ref = data.get_train();
        let num_labels = data.num_labels();
        let size = data.train_size();
        let mut inputs = match num_labels == 0 {
            true => {
                vec![Vec::with_capacity(size); 1]
            }
            false => {
                vec![Vec::with_capacity(size); num_labels]
            }
        };

        let option_ref = data_ref.0.as_ref();
        for i in 0..size {
            if num_labels == 0 {
                inputs[0].push(data_ref.1[i].clone());
            } else {
                let target = option_ref.map_or(0, |val| val[i]);
                inputs[target].push(data_ref.1[i].clone());
            }
        }

        inputs
    }

    pub fn new<T>(inputs: &T) -> Self
    where
        T: FileReader,
    {
        let inputs = Self::format_input_data(inputs);
        let num_labels = inputs.len();
        let num_attributes = inputs[0][0].len();
        let mut state = Vec::with_capacity(num_attributes);
        let size = inputs.iter().map(|label| label.len()).sum::<usize>();
        let mut initial_state = Vec::new();
        for input in inputs.iter() {
            initial_state.push((0..input.len()).collect::<Vec<usize>>())
        }
        state.push(initial_state);

        let mut structure = Horizontal {
            input: inputs,
            support: size,
            labels_support: Vec::with_capacity(num_labels),
            num_labels,
            num_attributes,
            position: Vec::with_capacity(num_attributes),
            state,
        };
        structure.support();
        structure
    }

    fn get_last_state(&self) -> Option<&Vec<Vec<usize>>> {
        self.state.last()
    }

    fn pushing(&mut self, item: usize) {
        let mut new_state = Vec::new();
        self.support = 0;
        self.labels_support.clear();
        for _ in 0..self.num_labels {
            self.labels_support.push(0);
        }
        if let Some(last) = self.state.last() {
            for (i, label_state) in last.iter().enumerate() {
                let mut label_transactions = Vec::with_capacity(label_state.len());
                for transaction in label_state {
                    let input = &self.input[i][*transaction];
                    let feature = attribute(item);
                    let it_type = item_type(item);
                    if input[feature] == it_type {
                        label_transactions.push(*transaction);
                    }
                }
                self.support += label_transactions.len();
                if self.num_labels > 0 {
                    self.labels_support[i] = label_transactions.len();
                }
                new_state.push(label_transactions);
            }
        }

        self.state.push(new_state);
    }
}

#[cfg(test)]
mod test_horizontal_binary_structure {
    use crate::data::BinaryData;
    use crate::data::FileReader;
    use crate::globals::item;
    use crate::structures::{Horizontal, Structure};

    #[test]
    fn load_horizontal_structure() {
        let dataset = BinaryData::read("test_data/small.txt", false, 0.0);
        let data_structure = Horizontal::new(&dataset);
        let state = [[[0usize, 1], [0, 1]]];
        let input = [[[1usize, 0, 1], [0, 1, 1]], [[0, 0, 0], [0, 1, 0]]];

        assert_eq!(data_structure.position.len(), 0);
        assert_eq!(data_structure.num_labels(), 2);
        assert_eq!(data_structure.state.iter().eq(state.iter()), true);
        assert_eq!(data_structure.input.iter().eq(input.iter()), true);
        assert_eq!(data_structure.label_support(0), 2);
        assert_eq!(data_structure.label_support(1), 2);
    }

    #[test]
    fn moving_one_step() {
        let dataset = BinaryData::read("test_data/small.txt", false, 0.0);
        let mut data_structure = Horizontal::new(&dataset);
        let position = [item(0usize, 0usize)];
        let true_state = vec![vec![1usize], vec![0, 1]];

        data_structure.push(item(0, 0));
        assert_eq!(data_structure.position.iter().eq(position.iter()), true);
        assert_eq!(data_structure.support, 3);
        assert_eq!(data_structure.label_support(0), 1);
        assert_eq!(data_structure.label_support(1), 2);

        let state = data_structure.get_last_state();
        if let Some(state) = state {
            assert_eq!(state.iter().eq(true_state.iter()), true);
        }
    }
    #[test]
    fn backtracking() {
        let dataset = BinaryData::read("test_data/small.txt", false, 0.0);
        let mut data_structure = Horizontal::new(&dataset);

        let position = [item(2usize, 1usize), item(0, 1)];
        let real_state = vec![vec![0usize], vec![]];

        data_structure.push(item(2, 1));
        data_structure.push(item(0, 1));
        assert_eq!(data_structure.position.len(), 2);
        assert_eq!(data_structure.position.iter().eq(position.iter()), true);
        assert_eq!(data_structure.support, 1);
        assert_eq!(data_structure.label_support(0), 1);
        assert_eq!(data_structure.label_support(1), 0);
        let state = data_structure.get_last_state();
        if let Some(state) = state {
            assert_eq!(state.iter().eq(real_state.iter()), true);
        }

        data_structure.backtrack();
        let position = [position[0]];
        assert_eq!(data_structure.position.len(), 1);
        assert_eq!(data_structure.position.iter().eq(position.iter()), true);
        assert_eq!(data_structure.support(), 2);
        assert_eq!(data_structure.label_support(0), 2);
        assert_eq!(data_structure.label_support(1), 0);

        let real_state = vec![vec![0usize, 1], vec![]];

        let state = data_structure.get_last_state();
        if let Some(state) = state {
            assert_eq!(state.iter().eq(real_state.iter()), true);
        }
    }
}
