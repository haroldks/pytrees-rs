use crate::data::FileReader;
use crate::globals::{attribute, item_type};
use crate::structures::types::DoublePointerData;
use crate::structures::{DataCover, Difference, Structure};
use search_trail::{
    BoolManager, ReversibleBool, ReversibleUsize, SaveAndRestore, StateManager, UsizeManager,
};
use std::collections::HashSet;

struct State(usize, usize, usize, bool, usize);

// Contains the item,  start, middle and end if it is left.
// The idea is if the past is left (odd number) and the next is right (even number) and + 1 of left we can restore the state and just push the right item
#[derive(Debug)]
struct PastState(usize, [usize; 3], bool, usize);

impl PastState {
    fn update(&mut self, item: usize, state: [usize; 3], is_left: bool, support: usize) {
        self.0 = item;
        self.1 = state;
        self.2 = is_left;
        self.3 = support;
    }

    fn new() -> Self {
        Self(usize::MAX, [usize::MAX; 3], false, usize::MAX)
    }
}

pub struct DoublePointer {
    input: DoublePointerData,
    tids: Vec<usize>,
    support: usize,
    num_labels: usize,
    num_attributes: usize,
    labels_support: Vec<usize>,
    position: Vec<usize>,
    state: [ReversibleUsize; 3],
    is_left: ReversibleBool,
    distance: ReversibleUsize, // Steps to restore to attain the initial state
    manager: StateManager,
    past: PastState,
}

impl Structure for DoublePointer {
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
        let (start, end) = self.get_borders();
        let mut support = 0;
        for tid in self.tids[start..end].iter() {
            if self.input.target.as_ref().map_or(0, |target| target[*tid]) == label {
                support += 1;
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

        for _label in 0..self.num_labels {
            self.labels_support.push(0);
        }

        // Getting the concerned border for the current state
        let (start, end) = self.get_borders();

        // Looping over elements between the borders in tid vector
        for tid in self.tids[start..end].iter() {
            let label = self.input.target.as_ref().map_or(0, |target| target[*tid]);
            self.labels_support[label] += 1;
        }
        &self.labels_support
    }

    fn support(&mut self) -> usize {
        if !self.support == usize::MAX {
            return self.support;
        }
        let (start, end) = self.get_borders();
        if start >= end {
            self.support = 0;
            return 0;
        }
        self.support = end - start;
        self.support
    }

    fn get_support(&self) -> usize {
        self.support
    }

    fn push(&mut self, item: usize) -> usize {
        self.position.push(item);
        let current_distance = self.manager.get_usize(self.distance);
        self.manager.set_usize(self.distance, current_distance + 1);
        self.manager.save_state();

        // Reading the past state
        let past_item = self.past.0;

        // This means that the past item is left and the current item is right
        if past_item % 2 == 0 && item == past_item + 1 {
            // We get the parent support
            let parent_support = self.support();

            // We restore the state
            assert!(self.past.2);
            self.manager.set_usize(self.state[0], self.past.1[0]);
            self.manager.set_usize(self.state[1], self.past.1[1]);
            self.manager.set_usize(self.state[2], self.past.1[2]);
            self.manager.set_bool(self.is_left, false);
            self.support = parent_support - self.past.3;
            return self.support;
        }

        let statue_value = self.pushing(item);
        self.push_state(statue_value);
        self.support
    }

    fn backtrack(&mut self) {
        if !self.position.is_empty() {
            let past_item = self.position.pop().unwrap();
            let state_value = self.get_state();
            let is_left = self.manager.get_bool(self.is_left);
            self.past
                .update(past_item, state_value, is_left, self.support);
            self.manager.restore_state();
            self.labels_support.clear();
            self.support();
        }
    }

    fn temp_push(&mut self, item: usize) -> usize {
        let statue_value = self.pushing(item);
        statue_value.4
    }

    fn reset(&mut self) {
        self.position.clear();
        let distance = self.manager.get_usize(self.distance);
        for _ in 0..distance + 1 {
            self.manager.restore_state();
        }
        self.support = <usize>::MAX;
        self.labels_support.clear();
    }

    fn get_position(&self) -> &[usize] {
        &self.position
    }

    fn get_data_cover(&mut self) -> DataCover {
        let mut cover = vec![];
        if self.position.is_empty() {
            cover = self.tids.iter().map(|tid| *tid as u64).collect()
        }
        let (start, end) = self.get_borders();
        cover = self.tids[start..end]
            .iter()
            .map(|tid| *tid as u64)
            .collect();

        DataCover {
            cover,
            support: self.support(),
            ..DataCover::default()
        }
    }

    fn get_difference(&self, data_cover: &DataCover) -> Difference {
        let (start, end) = self.get_borders();

        let cover = self.tids[start..end]
            .iter()
            .map(|tid| *tid)
            .collect::<HashSet<usize>>();
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
        if self.position.is_empty() {
            return self.tids.clone();
        }
        let (start, end) = self.get_borders();
        self.tids[start..end].to_vec()
    }
}

impl DoublePointer {
    pub fn format_input_data<T>(data: &T) -> DoublePointerData
    // TODO: Cancel cloning
    where
        T: FileReader,
    {
        let data_ref = data.get_train();

        let target = data_ref.0.clone();
        let num_labels = data.num_labels();
        let num_attributes = data.num_attributes();
        let mut inputs = vec![Vec::with_capacity(data.train_size()); num_attributes];
        for row in data_ref.1.iter() {
            for (i, val) in row.iter().enumerate() {
                inputs[i].push(*val);
            }
        }

        DoublePointerData {
            inputs,
            target,
            num_labels,
            num_attributes,
        }
    }

    pub fn new<T>(input: &T) -> Self
    where
        T: FileReader,
    {
        let input = Self::format_input_data(input);
        let support = input.inputs[0].len();
        let tids = (0..support).collect::<Vec<usize>>();
        let mut manager = StateManager::default();
        let state = [
            manager.manage_usize(0),
            manager.manage_usize(tids.len()),
            manager.manage_usize(tids.len()),
        ];
        let is_left = manager.manage_bool(true);
        let distance = manager.manage_usize(0);

        manager.save_state(); // Save the initial state
        let num_labels = input.num_labels;
        let num_attributes = input.num_attributes;

        Self {
            input,
            tids,
            support,
            num_labels,
            num_attributes,
            labels_support: Vec::with_capacity(num_labels),
            position: vec![],
            state,
            is_left,
            distance,
            manager,
            past: PastState::new(),
        }
    }

    fn pushing(&mut self, item: usize) -> State {
        self.labels_support.clear();
        let (start, end) = self.get_borders();
        let attribute = attribute(item);
        let is_left = item_type(item) == 0;
        let mut i = start;
        let mut j = end;
        loop {
            while i < j && self.input.inputs[attribute][self.tids[i]] == 0 {
                i += 1
            }

            if i + 1 >= j {
                break;
            }

            while j > i {
                j -= 1; // Decrements j before checking
                if self.input.inputs[attribute][self.tids[j]] != 1 {
                    break;
                }
            }

            if i == j {
                break;
            }

            self.tids.swap(i, j);
            i += 1;
        }

        let mut support = 0;
        if is_left {
            support = i - start;
        } else {
            support = end - i;
        }
        State(start, i, end, is_left, support)
    }

    fn push_state(&mut self, state_value: State) {
        self.manager.set_usize(self.state[0], state_value.0);
        self.manager.set_usize(self.state[1], state_value.1);
        self.manager.set_usize(self.state[2], state_value.2);
        self.manager.set_bool(self.is_left, state_value.3);
        self.support = state_value.4;
    }

    fn get_borders(&self) -> (usize, usize) {
        let is_left = self.manager.get_bool(self.is_left);
        match is_left {
            true => (
                self.manager.get_usize(self.state[0]),
                self.manager.get_usize(self.state[1]),
            ),
            false => (
                self.manager.get_usize(self.state[1]),
                self.manager.get_usize(self.state[2]),
            ), // FIXME: Check if this is correct
        }
    }
    pub fn get_state(&self) -> [usize; 3] {
        let state = &self.state;
        let mut state_value = [0; 3];
        for (i, elem) in state.iter().enumerate() {
            state_value[i] = self.manager.get_usize(*elem);
        }
        state_value
    }

    pub fn set_state(&mut self, state_value: &[usize; 3], item: usize) {
        self.position.push(item);
        self.support = <usize>::MAX;
        let current_distance = self.manager.get_usize(self.distance);
        self.manager.set_usize(self.distance, current_distance + 1);
        self.manager.save_state();
        let is_left = item_type(item) == 0;
        let state = &self.state;
        for (i, elem) in state.iter().enumerate() {
            self.manager.set_usize(*elem, state_value[i]);
        }
        self.manager.set_bool(self.is_left, is_left);
    }
}

#[cfg(test)]
mod test_double_pointer {
    use crate::data::binary_data::BinaryData;
    use crate::data::FileReader;
    use crate::globals::item;
    use crate::structures::{DoublePointer, Structure};
    use search_trail::{BoolManager, UsizeManager};

    #[test]
    fn read_double_pointer() {
        let dataset = BinaryData::read("test_data/small.txt", false, 0.0);
        let bitset_data = DoublePointer::format_input_data(&dataset);
        let data = [[1usize, 0, 0, 0], [0, 1, 0, 1], [1, 1, 0, 0]];
        let target = [0usize, 0, 1, 1];
        assert_eq!(bitset_data.inputs.iter().eq(data.iter()), true);
        assert_eq!(
            bitset_data
                .target
                .map_or(false, |ref t| { t.iter().eq(target.iter()) }),
            true
        );
        assert_eq!(bitset_data.num_labels, 2);
        assert_eq!(bitset_data.num_attributes, 3);
    }

    #[test]
    fn checking_inside_values() {
        let dataset = BinaryData::read("test_data/small.txt", false, 0.0);
        let mut structure = DoublePointer::new(&dataset);
        assert_eq!(structure.num_labels(), 2);
        assert_eq!(structure.num_attributes(), 3);
        assert_eq!(structure.tids, [0, 1, 2, 3]);

        // Getting manager
        let manager = &structure.manager;
        let state = &structure.state;
        let is_left = &structure.is_left;
        let pos_value = manager.get_bool(*is_left);
        assert_eq!(pos_value, true);
        let start = manager.get_usize(state[0]);
        let middle = manager.get_usize(state[1]);
        let end = manager.get_usize(state[2]);
        assert_eq!(start, 0);
        assert_eq!(middle, 4);
        assert_eq!(end, 4);
        assert_eq!(structure.position.is_empty(), true);
    }
    #[test]
    fn checking_root_data_structure() {
        let dataset = BinaryData::read("test_data/small_.txt", false, 0.0);
        let mut structure = DoublePointer::new(&dataset);
        assert_eq!(structure.support(), 10);
        assert_eq!(structure.label_support(0), 5);
        assert_eq!(structure.label_support(1), 5);
        assert_eq!(structure.labels_support(), &[5, 5]);
    }

    #[test]
    fn moving_from_root_once() {
        let dataset = BinaryData::read("test_data/small.txt", false, 0.0);
        let mut structure = DoublePointer::new(&dataset);

        let zero_item = item(0, 0);
        structure.push(zero_item);

        assert_eq!(structure.tids, [3, 1, 2, 0]);
        let manager = &structure.manager;
        let state = &structure.state;
        let mut state_value = vec![];
        let is_left = manager.get_bool(structure.is_left);
        assert_eq!(is_left, true);

        for elem in state.iter() {
            state_value.push(manager.get_usize(*elem));
        }
        assert_eq!(state_value, [0, 3, 4]);

        assert_eq!(structure.support(), 3);
        assert_eq!(structure.label_support(0), 1);
        assert_eq!(structure.label_support(1), 2);
        assert_eq!(structure.labels_support(), &[1, 2]);
    }

    #[test]
    fn moving_from_root_more_than_once() {
        let dataset = BinaryData::read("test_data/small.txt", false, 0.0);
        let mut structure = DoublePointer::new(&dataset);

        structure.push(item(0, 0));

        structure.push(item(1, 0));

        assert_eq!(structure.tids, [2, 1, 3, 0]);

        let mut support = structure.support();
        assert_eq!(support, 1);
        assert_eq!(structure.label_support(0), 0);
        assert_eq!(structure.label_support(1), 1);
        assert_eq!(structure.labels_support(), &[0, 1]);
        let state = structure.get_state();

        structure.push(item(2, 0));
        assert_eq!(structure.tids, [2, 1, 3, 0]);

        support = structure.support();
        let state = structure.get_state();

        assert_eq!(support, 1);
        assert_eq!(structure.label_support(0), 0);
        assert_eq!(structure.label_support(1), 1);
        assert_eq!(structure.labels_support(), &[0, 1]);
    }

    #[test]
    fn moving_towards_null_support() {
        let dataset = BinaryData::read("test_data/small.txt", false, 0.0);
        let mut structure = DoublePointer::new(&dataset);

        structure.push(item(0, 0));

        let support = structure.push(item(1, 1));

        assert_eq!(support, 2);
        assert_eq!(structure.label_support(0), 1);
        assert_eq!(structure.label_support(1), 1);
        assert_eq!(structure.labels_support(), &[1, 1]);

        let support = structure.push(item(2, 0));

        assert_eq!(support, 1);
    }

    #[test]
    fn testing_temp_push() {
        let dataset = BinaryData::read("test_data/anneal.txt", false, 0.0);
        let mut structure = DoublePointer::new(&dataset);
        let num_attributes = structure.num_attributes();

        assert_eq!(
            structure.labels_support().iter().eq([187, 625].iter()),
            true
        );
        assert_eq!(structure.temp_push(item(43, 1)), 26);
        assert_eq!(
            structure.labels_support().iter().eq([187, 625].iter()),
            true
        );
        assert_eq!(structure.temp_push(item(43, 0)), 786);
        assert_eq!(
            structure.labels_support().iter().eq([187, 625].iter()),
            true
        );
    }

    #[test]
    fn test_backtracking() {
        let dataset = BinaryData::read("test_data/small_.txt", false, 0.0);
        let mut structure = DoublePointer::new(&dataset);

        let support = structure.push(item(3, 1));
        let s = structure.temp_push(item(2, 1));
        assert_eq!(s, 2);

        structure.push(item(0, 0));
        structure.backtrack();

        let expected_position = [item(3, 1usize)];

        assert_eq!(structure.position.iter().eq(expected_position.iter()), true);
        assert_eq!(structure.support, support);
    }
    #[test]
    fn moving_on_step_and_backtrack() {
        let dataset = BinaryData::read("test_data/small_.txt", false, 0.0);
        let mut structure = DoublePointer::new(&dataset);

        let num_attributes = structure.num_attributes;
        let expected_supports = [5usize, 5, 6, 6];

        for i in 0..num_attributes {
            let support = structure.push(item(i, 0));
            structure.backtrack();
            assert_eq!(support, expected_supports[i]);
        }
    }
}
