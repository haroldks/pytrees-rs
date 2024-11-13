// * Done
use crate::data::FileReader;
use crate::globals::{attribute, item_type};
use crate::structures::types::BitsetStructData;
use crate::structures::{format_data_into_bitset, DataCover, Difference, Structure};
use search_trail::{ReversibleU64, SaveAndRestore, StateManager, U64Manager};

pub struct RevBitset {
    inputs: BitsetStructData,
    support: usize,
    labels_support: Vec<usize>,
    num_labels: usize,
    num_attributes: usize,
    position: Vec<usize>,
    state_manager: StateManager,
    state: Vec<ReversibleU64>,
    index: Vec<usize>,
    limit: Vec<isize>,
    distance: ReversibleU64, // Steps to restore to attain the initial state
}

impl Default for RevBitset {
    fn default() -> Self {
        let mut fake_manager = StateManager::default();
        let distance = fake_manager.manage_u64(0);

        Self {
            inputs: Default::default(),
            support: 0,
            labels_support: vec![],
            num_labels: 0,
            num_attributes: 0,
            position: vec![],
            state_manager: fake_manager,
            state: vec![],
            index: vec![],
            limit: vec![],
            distance,
        }
    }
}

impl Structure for RevBitset {
    fn num_attributes(&self) -> usize {
        self.num_attributes
    }

    fn num_labels(&self) -> usize {
        self.num_labels
    }

    fn label_support(&self, label: usize) -> usize {
        // FIXME: Useless
        let state = &self.state;
        let support = usize::MAX;

        if label < self.num_labels {
            if let Some(limit) = self.limit.last() {
                let mut count = 0;
                if *limit >= 0 {
                    let label_bitset = &self.inputs.targets[label];
                    for i in 0..(*limit + 1) as usize {
                        let cursor = self.index[i];
                        let val = self.state_manager.get_u64(state[cursor]);
                        count += (label_bitset[cursor] & val).count_ones()
                    }
                }
                return count as usize;
            }
        }
        support
    }

    fn labels_support(&mut self) -> &[usize] {
        if !self.labels_support.is_empty() {
            return &self.labels_support;
        }

        self.labels_support.clear();
        for _ in 0..self.num_labels {
            self.labels_support.push(0);
        }

        if let Some(limit) = self.limit.last() {
            if self.num_labels == 2 {
                if *limit >= 0 {
                    let label_bitset = &self.inputs.targets[0];
                    let mut count = 0;
                    for i in 0..(*limit + 1) as usize {
                        let cursor = self.index[i];
                        let val = self.state_manager.get_u64(self.state[cursor]);
                        count += (label_bitset[cursor] & val).count_ones()
                    }
                    self.labels_support[0] = count as usize;
                    self.labels_support[1] = self.support() - count as usize;
                }
                return &self.labels_support;
            }

            for label in 0..self.num_labels {
                let mut count = 0;
                if *limit >= 0 {
                    let label_bitset = &self.inputs.targets[label];
                    for i in 0..(*limit + 1) as usize {
                        let cursor = self.index[i];
                        let val = self.state_manager.get_u64(self.state[cursor]);
                        count += (label_bitset[cursor] & val).count_ones()
                    }
                }
                self.labels_support[label] = count as usize;
            }
            return &self.labels_support;
        }
        &self.labels_support
    }

    fn support(&mut self) -> usize {
        if !self.support == usize::MAX {
            return self.support;
        }
        self.support = 0;
        if let Some(limit) = self.limit.last() {
            if *limit >= 0 {
                for i in 0..(*limit + 1) as usize {
                    let cursor = self.index[i];
                    let val = self.state_manager.get_u64(self.state[cursor]);
                    self.support += val.count_ones() as usize;
                }
            }
        }
        self.support
    }

    fn get_support(&self) -> usize {
        self.support
    }

    fn push(&mut self, item: usize) -> usize {
        self.position.push(item);
        let current_distance = self.state_manager.get_u64(self.distance);
        self.state_manager
            .set_u64(self.distance, current_distance + 1);
        self.state_manager.save_state();
        self.pushing(item);

        self.support()
    }

    fn backtrack(&mut self) {
        // TODO: Remove the support computation
        if !self.position.is_empty() {
            self.position.pop();
            let limit_size = self.limit.len();
            if self.is_empty() && limit_size > 1 && self.limit[limit_size - 2] < 0 {
                self.limit.pop();
            } else if let Some(_limit) = self.limit.last() {
                self.state_manager.restore_state();
                self.limit.pop();
            }
            self.support = usize::MAX;
            self.labels_support.clear();
            // self.support();
        }
    }

    fn temp_push(&mut self, item: usize) -> usize {
        // TODO: Change this to avoid recomputing the support & labels support
        let mut support = 0;
        if let Some(limit) = self.limit.last() {
            let limit = *limit;
            if limit >= 0 {
                let feature = attribute(item);
                let value = item_type(item);
                let feature_vec = &self.inputs.inputs[feature];
                let lim = limit as usize;
                for i in (0..lim + 1).rev() {
                    let cursor = self.index[i];
                    let val = self.state_manager.get_u64(self.state[cursor]);
                    let word = match value {
                        0 => val & !feature_vec[cursor],
                        _ => val & feature_vec[cursor],
                    };
                    let word_count = word.count_ones() as usize;
                    support += word_count;
                }
            }
        }
        support
    }

    fn reset(&mut self) {
        self.position = Vec::with_capacity(self.num_attributes);
        self.limit = Vec::with_capacity(self.num_attributes);
        self.limit.push((self.inputs.chunks - 1) as isize);
        let distance = self.state_manager.get_u64(self.distance);
        for _ in 0..distance + 1 {
            self.state_manager.restore_state();
        }
        self.support = self.inputs.size;
        self.labels_support.clear();
    }

    fn get_position(&self) -> &[usize] {
        &self.position
    }

    fn get_data_cover(&mut self) -> DataCover {
        let mut current_state = vec![0; self.inputs.chunks];
        let mut data_cover = DataCover::default();
        if let Some(limit) = self.limit.last() {
            if *limit >= 0 {
                for i in 0..(*limit + 1) {
                    let cursor = self.index[i as usize];
                    let word = self.state_manager.get_u64(self.state[cursor]);
                    current_state[cursor] = word;
                }

                data_cover = DataCover {
                    cover: current_state,
                    limit: *limit,
                    index: self.index.clone(),
                    support: self.support(),
                    ..DataCover::default()
                }
            }
        }
        data_cover
    }

    fn get_difference(&self, data_cover: &DataCover) -> Difference {
        let current_limit = self.get_current_limit();
        let data_cover_limit = data_cover.limit;
        let current_index = &self.index;
        let data_cover_index = &data_cover.index;
        let mut in_count = 0;
        let mut out_count = 0;
        let mut data_in = true;
        for (limit, index) in [current_limit, data_cover_limit]
            .iter()
            .zip([current_index, data_cover_index])
        {
            if *limit >= 0 {
                for cursor in index.iter().take(*limit as usize + 1) {
                    let val = match current_limit == -1 {
                        true => 0,
                        false => self.state_manager.get_u64(self.state[*cursor]),
                    };
                    match data_in {
                        true => in_count += (val & !data_cover.cover[*cursor]).count_ones(),
                        false => out_count += (data_cover.cover[*cursor] & !val).count_ones(),
                    };
                }
            }
            data_in = false;
        }
        (in_count as usize, out_count as usize)
    }

    fn get_tids(&self) -> Vec<usize> {
        if self.position.is_empty() {
            return (0..self.inputs.size).collect::<Vec<usize>>();
        }
        let mut tids = Vec::with_capacity(self.inputs.size);
        let nb_chunks = self.inputs.chunks;
        let nb_trans = self.inputs.size;
        if let Some(limit) = self.limit.last() {
            if *limit >= 0 {
                let limit = *limit as usize;
                for i in 0..limit + 1 {
                    let cursor = self.index[i];
                    let mut word = self.state_manager.get_u64(self.state[cursor]);
                    while word != 0 {
                        let set_bits = word.trailing_zeros() as usize;
                        let tid = nb_trans - ((nb_chunks - 1 - cursor) * 64 + set_bits) - 1;
                        tids.push(tid);
                        word &= !(1 << set_bits);
                    }
                }
            }
        }
        tids
    }
}

impl RevBitset {
    pub fn new<T>(inputs: &T) -> RevBitset
    where
        T: FileReader,
    {
        let inputs = format_data_into_bitset(inputs);
        let index = (0..inputs.chunks).collect::<Vec<usize>>();
        let num_attributes = inputs.inputs.len();
        let mut state = Vec::with_capacity(inputs.chunks);
        let mut manager = StateManager::default();
        for _ in 0..inputs.chunks {
            let n = manager.manage_u64(u64::MAX);
            state.push(n);
        }

        if inputs.size % 64 != 0 {
            let first_dead_bit = 64 - (inputs.chunks * 64 - inputs.size);
            let first_chunk = &mut state[0];

            let mut val = manager.get_u64(*first_chunk);
            for i in (first_dead_bit..64).rev() {
                let int_mask = 1u64 << i;
                val &= !int_mask;
            }
            manager.set_u64(*first_chunk, val);
        }

        let distance = manager.manage_u64(0);

        manager.save_state(); // Save the initial state of the manager

        let mut limit = Vec::with_capacity(num_attributes);
        limit.push((inputs.chunks - 1) as isize);
        let support = inputs.size;
        let num_labels = inputs.targets.len();

        let mut structure = RevBitset {
            inputs,
            support,
            labels_support: Vec::with_capacity(num_labels),
            num_labels,
            num_attributes,
            position: Vec::with_capacity(num_attributes),
            state_manager: manager,
            state,
            index,
            limit,
            distance,
        };
        structure.support();
        structure
    }

    fn pushing(&mut self, item: usize) {
        self.support = 0;
        self.labels_support.clear();

        for _ in 0..self.num_labels {
            self.labels_support.push(0);
        }

        if let Some(limit) = self.limit.last() {
            let mut limit = *limit;
            if limit >= 0 {
                let feature = attribute(item);
                let value = item_type(item);
                let feature_vec = &self.inputs.inputs[feature];
                let mut lim = limit as usize;
                for i in (0..lim + 1).rev() {
                    let cursor = self.index[i];
                    let val = self.state_manager.get_u64(self.state[cursor]);
                    let word = match value {
                        0 => val & !feature_vec[cursor],
                        _ => val & feature_vec[cursor],
                    };
                    if word == 0 {
                        self.index[i] = self.index[lim];
                        self.index[lim] = cursor;
                        limit -= 1;
                        lim = lim.saturating_sub(1);
                        if limit < 0 {
                            break;
                        }
                    } else {
                        let word_count = word.count_ones() as usize;
                        self.support += word_count;
                        if self.num_labels == 2 {
                            let label_val = &self.inputs.targets[0][cursor];
                            let zero_count = (label_val & word).count_ones() as usize;
                            self.labels_support[0] += zero_count;
                            self.labels_support[1] += word_count - zero_count;
                        } else {
                            for j in 0..self.num_labels {
                                let label_val = &self.inputs.targets[j][cursor];
                                self.labels_support[j] += (label_val & word).count_ones() as usize;
                            }
                        }

                        self.state_manager.set_u64(self.state[cursor], word);
                    }
                }
            }

            self.limit.push(limit);
        }
    }

    fn is_empty(&self) -> bool {
        if let Some(limit) = self.limit.last() {
            return *limit < 0;
        }
        false
    }

    pub fn get_current_limit(&self) -> isize {
        self.limit.last().copied().unwrap_or(-1)
    }
}

#[cfg(test)]
mod test_trail {
    use crate::data::binary_data::BinaryData;
    use crate::data::FileReader;
    use crate::globals::item;
    use crate::structures::{RevBitset, Structure};

    #[test]
    fn test_trail_stats() {
        let dataset = BinaryData::read("test_data/rsparse_dataset.txt", false, 0.0);
        let structure = RevBitset::new(&dataset);

        let expected_support = 192;
        assert_eq!(structure.support, expected_support);

        let expected_label_supports = [64usize, 128];
        assert_eq!(structure.label_support(0), expected_label_supports[0]);
        assert_eq!(structure.label_support(1), expected_label_supports[1]);
    }

    #[test]
    fn test_trail_branching_in_data() {
        let dataset = BinaryData::read("test_data/rsparse_dataset.txt", false, 0.0);
        let mut structure = RevBitset::new(&dataset);

        let support = structure.push(item(0, 1));

        assert_eq!(support, 128);
        if let Some(limit) = structure.limit.last() {
            assert_eq!(*limit, 1);
        }

        assert_eq!(structure.index.iter().eq([0, 2, 1].iter()), true);
        assert_eq!(structure.label_support(1), 128);
        assert_eq!(structure.label_support(0), 0);

        let support = structure.push(item(1, 0));

        assert_eq!(support, 64);
        if let Some(limit) = structure.limit.last() {
            assert_eq!(*limit, 0);
        }
        assert_eq!(structure.index.iter().eq([0, 2, 1].iter()), true);
        assert_eq!(structure.label_support(1), 64);
        assert_eq!(structure.label_support(0), 0);

        structure.backtrack();

        assert_eq!(structure.support(), 128);
        if let Some(limit) = structure.limit.last() {
            assert_eq!(*limit, 1);
        }

        assert_eq!(structure.index.iter().eq([0, 2, 1].iter()), true);
        assert_eq!(structure.label_support(1), 128);
        assert_eq!(structure.label_support(0), 0);
    }

    #[test]
    fn compute_state_on_small_dataset_with_trail() {
        let dataset = BinaryData::read("test_data/small.txt", false, 0.0);
        let mut structure = RevBitset::new(&dataset);
        let num_attributes = structure.num_attributes();

        let support = structure.push(item(0, 1));
        assert_eq!(support, 1);
        assert_eq!(structure.label_support(0), 1);
        assert_eq!(structure.label_support(1), 0);
        assert_eq!(structure.labels_support().iter().eq([1, 0].iter()), true);

        let support = structure.push(item(1, 1));
        assert_eq!(structure.is_empty(), true);
        assert_eq!(structure.label_support(0), 0);
        assert_eq!(structure.label_support(1), 0);

        structure.push(item(2, 1));

        assert_eq!(structure.limit.iter().eq([0, 0, -1, -1].iter()), true);

        structure.backtrack();
        assert_eq!(structure.limit.iter().eq([0, 0, -1].iter()), true);

        structure.backtrack();
        assert_eq!(structure.limit.iter().eq([0, 0].iter()), true);
        assert_eq!(structure.support(), 1);
        assert_eq!(structure.label_support(0), 1);
        assert_eq!(structure.label_support(1), 0);
        assert_eq!(structure.labels_support().iter().eq([1, 0].iter()), true);
    }

    #[test]
    fn test_temp_push_on_trail() {
        let dataset = BinaryData::read("test_data/anneal.txt", false, 0.0);
        let mut structure = RevBitset::new(&dataset);
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
    fn check_trail_reset() {
        let dataset = BinaryData::read("test_data/anneal.txt", false, 0.0);
        let mut structure = RevBitset::new(&dataset);

        for i in 0..structure.num_attributes() / 4 {
            &mut structure.push(item(i, 0));
        }

        structure.reset();

        assert_eq!(structure.support(), 812);
        assert_eq!(
            structure.labels_support().iter().eq([187, 625].iter()),
            true
        );
    }

    #[test]
    fn check_double_backtrack() {
        let dataset = BinaryData::read("test_data/anneal.txt", false, 0.0);
        let mut structure = RevBitset::new(&dataset);

        // Root
        println!("Limits : {:?}", structure.limit);
        println!("Index : {:?}", structure.index);
        println!("Support {:?}", structure.support());
        println!("Label support {:?}", structure.labels_support());

        // (4, 1)
        let support = structure.push(item(4, 1));
        println!("Limits : {:?}", structure.limit);
        println!("Index : {:?}", structure.index);
        println!("Support {:?}", support);
        println!("Label support {:?}", structure.labels_support());

        // (4, 1) (5, 1)
        let support = structure.push(item(5, 1));
        println!("Limits : {:?}", structure.limit);
        println!("Index : {:?}", structure.index);
        println!("Support {:?}", support);
        println!("Label support {:?}", structure.labels_support());

        // (4, 1)
        structure.backtrack();
        println!("Limits : {:?}", structure.limit);
        println!("Index : {:?}", structure.index);
        println!("bSupport {:?}", structure.support());
        println!("Label support {:?}", structure.labels_support());

        // Root
        structure.backtrack();
        println!("Limits : {:?}", structure.limit);
        println!("Index : {:?}", structure.index);
        println!("bSupport {:?}", structure.support());
        println!("Label support {:?}", structure.labels_support());

        // (5, 1)
        let support = structure.push(item(5, 1));
        println!("Limits : {:?}", structure.limit);
        println!("Index : {:?}", structure.index);
        println!("nSupport {:?}", support);
        println!("Label support {:?}", structure.labels_support());
    }
}
