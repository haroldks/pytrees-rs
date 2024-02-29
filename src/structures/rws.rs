use crate::data::{BinaryData, FileReader};
use crate::globals::{attribute, item_type};
use crate::structures::{DataCover, Difference, Structure};
use std::collections::HashSet;

#[derive(Clone)]
pub struct RawBinary {
    input: BinaryData,
    support: usize,
    labels_support: Vec<usize>,
    num_labels: usize,
    num_attributes: usize,
    position: Vec<usize>,
    state: Vec<Vec<usize>>,
}

impl Structure for RawBinary {
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
            let target = self.input.get_train().0.as_ref();
            if let Some(state) = self.get_last_state() {
                support = state
                    .iter()
                    .filter(|x| target.map_or(false, |val| val[**x] == label))
                    .count();
            }
        }
        support
    }

    fn labels_support(&mut self) -> &[usize] {
        if !self.labels_support.is_empty() {
            return &self.labels_support;
        }

        for label in 0..self.num_labels {
            self.labels_support.push(self.label_support(label))
        }
        &self.labels_support
    }

    fn support(&mut self) -> usize {
        if self.support < <usize>::MAX {
            return self.support;
        }
        if let Some(last) = self.get_last_state() {
            self.support = last.len();
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
            self.support = <usize>::MAX;
            self.labels_support.clear();
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
        self.state = state;
        self.position = Vec::with_capacity(self.num_attributes);
        self.support = self.input.size();
        self.labels_support.clear();
    }

    fn get_position(&self) -> &[usize] {
        &self.position
    }

    fn get_data_cover(&mut self) -> DataCover {
        let mut cover = vec![];
        if let Some(state) = self.state.last() {
            cover = state.iter().map(|tid| *tid as u64).collect();
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
            cover = state.iter().copied().collect::<HashSet<usize>>();
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
        self.state.last().unwrap().clone()
    }
}

impl RawBinary {
    pub fn new(inputs: &BinaryData) -> Self {
        let inputs = inputs.clone();
        let num_labels = inputs.num_labels();
        let num_attributes = inputs.num_attributes();
        let train_size = inputs.train_size();
        let initial_state = (0..train_size).collect::<Vec<usize>>();
        let mut state = Vec::with_capacity(inputs.num_attributes());
        state.push(initial_state);
        Self {
            input: inputs,
            support: <usize>::MAX,
            labels_support: Vec::with_capacity(num_labels),
            num_labels,
            num_attributes,
            position: Vec::with_capacity(num_attributes),
            state,
        }
    }

    fn get_last_state(&self) -> Option<&Vec<usize>> {
        self.state.last()
    }

    fn pushing(&mut self, item: usize) {
        let mut new_state = vec![];
        self.support = 0;
        self.labels_support.clear();
        for _ in 0..self.num_labels {
            self.labels_support.push(0);
        }
        if let Some(last) = self.state.last() {
            let inputs = &self.input.get_train().1;
            let target = self.input.get_train().0.as_ref();
            let feature = attribute(item);
            let it_type = item_type(item);
            for tid in last {
                if inputs[*tid][feature] == it_type {
                    new_state.push(*tid);
                    self.support += 1;
                    if self.num_labels > 0 {
                        let class = target.map_or(0, |val| val[*tid]);
                        self.labels_support[class] += 1;
                    }
                }
            }
        }
        self.state.push(new_state);
    }
}

#[cfg(test)]
mod test_raw_binary_structure {
    use crate::data::{BinaryData, FileReader};
    use crate::globals::item;
    use crate::structures::rws::RawBinary;
    use crate::structures::Structure;

    #[test]
    fn test() {
        let dataset = BinaryData::read("test_data/small_.txt", false, 0.0);
        let mut raw_structure = RawBinary::new(&dataset);
        let support = raw_structure.support();
    }

    #[test]
    fn moving_one_step() {
        let dataset = BinaryData::read("test_data/small_.txt", false, 0.0);
        let mut data_structure = RawBinary::new(&dataset);

        let position = [item(0usize, 0usize)];
        let true_state = vec![1, 3, 4, 5, 9];

        data_structure.push(item(0, 0));
        assert_eq!(data_structure.position.iter().eq(position.iter()), true);
        assert_eq!(data_structure.support, 5);
        assert_eq!(data_structure.label_support(0), 2);
        assert_eq!(data_structure.label_support(1), 3);

        let state = data_structure.get_last_state();
        if let Some(state) = state {
            assert_eq!(state.iter().eq(true_state.iter()), true);
        }
    }
    #[test]
    fn backtracking() {
        let dataset = BinaryData::read("test_data/small_.txt", false, 0.0);
        let mut data_structure = RawBinary::new(&dataset);

        let position = [item(2usize, 1usize), item(0, 1)];
        let real_state = vec![6, 7];

        data_structure.push(item(2, 1));
        data_structure.push(item(0, 1));
        assert_eq!(data_structure.position.len(), 2);
        assert_eq!(data_structure.position.iter().eq(position.iter()), true);
        assert_eq!(data_structure.support(), 2);
        assert_eq!(data_structure.label_support(0), 1);
        assert_eq!(data_structure.label_support(1), 1);
        let state = data_structure.get_last_state();
        if let Some(state) = state {
            assert_eq!(state.iter().eq(real_state.iter()), true);
        }

        data_structure.backtrack();
        let position = [position[0]];
        assert_eq!(data_structure.position.len(), 1);
        assert_eq!(data_structure.position.iter().eq(position.iter()), true);
        assert_eq!(data_structure.support(), 4);
        assert_eq!(data_structure.label_support(0), 3);
        assert_eq!(data_structure.label_support(1), 1);

        let real_state = vec![3, 4, 6, 7];

        let state = data_structure.get_last_state();
        if let Some(state) = state {
            assert_eq!(state.iter().eq(real_state.iter()), true);
        }
    }
}
