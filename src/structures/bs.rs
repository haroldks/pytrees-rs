// * Done
use crate::data::FileReader;
use crate::globals::{attribute, item_type};
use crate::structures::types::BitsetStructData;

use crate::structures::{format_data_into_bitset, DataCover, Difference, Structure};

pub struct Bitset {
    inputs: BitsetStructData,
    support: usize,
    labels_support: Vec<usize>,
    num_attributes: usize,
    num_labels: usize,
    position: Vec<usize>,
    state: Vec<Vec<u64>>,
}

impl Structure for Bitset {
    fn num_attributes(&self) -> usize {
        self.num_attributes
    }

    fn num_labels(&self) -> usize {
        self.num_labels
    }

    fn label_support(&self, label: usize) -> usize {
        let support = <usize>::MAX;
        if label < self.num_labels {
            if let Some(state) = self.get_last_state() {
                let mut count = 0;
                let label_bitset = &self.inputs.targets[label];
                for (i, label_chunk) in label_bitset.iter().enumerate() {
                    count += (*label_chunk & state[i]).count_ones();
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

        if self.num_labels == 2 {
            if let Some(state) = self.get_last_state() {
                let mut count = 0;
                let label_bitset = &self.inputs.targets[0];
                for (i, label_chunk) in label_bitset.iter().enumerate() {
                    count += (*label_chunk & state[i]).count_ones();
                }
                self.labels_support.push(count as usize);
                let support = self.support();
                self.labels_support.push(support - count as usize);
            }
            return &self.labels_support;
        }

        if let Some(state) = self.state.last() {
            for label in 0..self.num_labels {
                let mut count = 0;
                let label_bitset = &self.inputs.targets[label];
                for (i, label_chunk) in label_bitset.iter().enumerate() {
                    count += (*label_chunk & state[i]).count_ones();
                }
                self.labels_support.push(count as usize);
            }
            return &self.labels_support;
        }
        &self.labels_support
    }

    fn support(&mut self) -> usize {
        if self.support < usize::MAX {
            return self.support;
        }
        self.support = 0;
        if let Some(current_state) = self.get_last_state() {
            self.support = current_state
                .iter()
                .map(|long| long.count_ones())
                .sum::<u32>() as usize;
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
            self.support = usize::MAX;
            self.labels_support.clear();
        }
    }

    fn temp_push(&mut self, item: usize) -> usize {
        let support = self.push(item);
        self.backtrack();
        support
    }

    fn reset(&mut self) {
        self.position = Vec::with_capacity(self.num_attributes);
        let mut state = Vec::with_capacity(self.num_attributes);
        state.push(self.state[0].clone());
        self.state = state;
        self.support = self.inputs.size;
        self.labels_support.clear();
    }
    fn get_position(&self) -> &[usize] {
        &self.position
    }

    fn get_data_cover(&mut self) -> DataCover {
        let mut data_cover = DataCover::default();
        if let Some(state) = self.state.last() {
            data_cover = DataCover {
                cover: state.clone(),
                support: self.support(),
                ..DataCover::default()
            }
        }
        data_cover
    }

    fn get_difference(&self, data_cover: &DataCover) -> Difference {
        let mut in_count = 0;
        let mut out_count = 0;

        if let Some(state) = self.state.last() {
            for (current, saved) in state.iter().zip(&data_cover.cover) {
                in_count += (current & !saved).count_ones();
                out_count += (saved & !current).count_ones();
            }
        }
        (in_count as usize, out_count as usize)
    }

    fn get_tids(&self) -> Vec<usize> {
        if self.position.is_empty() {
            return (0..self.inputs.size).collect();
        }

        let mut tids = vec![];
        let nb_chunks = self.inputs.chunks;
        let nb_trans = self.inputs.size;
        if let Some(state) = self.get_last_state() {
            for (idx, chunk) in state.iter().enumerate().rev() {
                let mut word = *chunk;
                while word != 0 {
                    let set_bit = word.trailing_zeros() as usize;
                    let tid = nb_trans - ((nb_chunks - 1 - idx) * 64 + set_bit) - 1;
                    tids.push(tid);
                    word &= !(1u64 << set_bit);
                }
            }
        }
        tids
    }
}

// impl BitsetTrait for BitsetStructure {
//     fn extract_leaf_bitvector(
//         &mut self,
//         tree: &Tree<NodeData>,
//         index: Index,
//         position: &mut Vec<Item>,
//         collector: &mut Vec<LeafInfo>,
//     ) {
//         let mut left_index = 0;
//         let mut right_index = 0;
//         let mut attribute = None;
//         if let Some(node) = tree.get_node(index) {
//             left_index = node.left;
//             right_index = node.right;
//             attribute = node.value.test;
//         }
//
//         if left_index == right_index {
//             let mut error = <usize>::MAX;
//             if let Some(node) = tree.get_node(index) {
//                 error = node.value.error;
//             }
//
//             // Is leaf
//             collector.push(LeafInfo {
//                 index,
//                 position: position.clone(),
//                 bitset: self.get_last_state_bitset(),
//                 error,
//             }); // Bizarre ca
//                 // position.pop();
//         }
//
//         if left_index > 0 {
//             if let Some(left_node) = tree.get_node(left_index) {
//                 let item = (attribute.unwrap(), 0);
//                 position.push(item);
//                 self.push(item);
//                 self.extract_leaf_bitvector(tree, left_index, position, collector);
//                 position.pop();
//                 self.backtrack()
//             }
//         }
//
//         if right_index > 0 {
//             if let Some(right_node) = tree.get_node(right_index) {
//                 let item = (attribute.unwrap(), 1);
//                 position.push(item);
//                 self.push(item);
//                 self.extract_leaf_bitvector(tree, right_index, position, collector);
//                 position.pop();
//                 self.backtrack()
//             }
//         }
//     }
//
//     fn set_state(&mut self, state: &Bitset, position: &Position) {
//         self.position = position.clone();
//         self.state = BitsetStackState::with_capacity(self.num_attributes);
//         self.state.push(state.clone());
//         self.support = usize::MAX;
//         self.labels_support.clear();
//     }
// }

impl Bitset {
    pub fn new<T>(inputs: &T) -> Self
    where
        T: FileReader,
    {
        let inputs = format_data_into_bitset(inputs);
        let num_attributes = inputs.inputs.len();
        let mut state = Vec::with_capacity(num_attributes);
        let mut initial_state = vec![<u64>::MAX; inputs.chunks];

        if inputs.size % 64 != 0 {
            let first_dead_bit = 64 - (inputs.chunks * 64 - inputs.size);
            let first_chunk = &mut initial_state[0];

            for i in (first_dead_bit..64).rev() {
                let int_mask = 1u64 << i;
                *first_chunk &= !int_mask;
            }
        }
        let support = inputs.size;
        let num_attributes = inputs.inputs.len();
        let num_labels = inputs.targets.len();
        state.push(initial_state);

        Bitset {
            inputs,
            support,
            labels_support: Vec::with_capacity(num_labels),
            num_attributes,
            num_labels,
            position: Vec::with_capacity(num_attributes),
            state,
        }
    }

    fn get_last_state(&self) -> Option<&Vec<u64>> {
        self.state.last()
    }

    fn pushing(&mut self, item: usize) {
        let mut new_state = Vec::new();
        self.support = 0;
        self.labels_support.clear();
        for _ in 0..self.num_labels {
            self.labels_support.push(0);
        }

        if let Some(last_state) = self.state.last() {
            let feature = attribute(item);
            let feature_vec = &self.inputs.inputs[feature];
            for (i, long) in feature_vec.iter().enumerate() {
                let word = match item_type(item) {
                    0 => last_state[i] & !*long,
                    _ => last_state[i] & *long,
                };

                let word_count = word.count_ones() as usize;
                self.support += word_count;

                if self.num_labels == 2 {
                    let label_chunk = &self.inputs.targets[0][i];
                    let zero_count = (word & label_chunk).count_ones() as usize;
                    self.labels_support[0] += zero_count;
                    self.labels_support[1] += word_count - zero_count;
                } else {
                    for n in 0..self.num_labels {
                        let label_chunk = &self.inputs.targets[n][i];
                        let label_count = (word & label_chunk).count_ones() as usize;
                        self.labels_support[i] += label_count;
                    }
                }

                new_state.push(word);
            }
        }
        self.state.push(new_state)
    }
}

#[cfg(test)]
mod test_bitsets {
    use crate::data::BinaryData;
    use crate::data::FileReader;
    use crate::globals::item;
    use crate::structures::Structure;
    use crate::structures::{format_data_into_bitset, Bitset};

    #[test]
    fn build_bitset_data() {
        let dataset = BinaryData::read("test_data/small_.txt", false, 0.0);
        let bitset_data = format_data_into_bitset(&dataset);
    }

    #[test]
    fn read_bitset_data_on_simple_small() {
        let dataset = BinaryData::read("test_data/small.txt", false, 0.0);
        let bitset_data = format_data_into_bitset(&dataset);

        let expected_inputs = [[8u64], [5], [12]];
        assert_eq!(bitset_data.inputs.iter().eq(expected_inputs.iter()), true);

        let expected_targets = [[12u64], [3]];
        assert_eq!(bitset_data.targets.iter().eq(expected_targets.iter()), true);

        assert_eq!(bitset_data.chunks, 1);
    }

    #[test]
    fn read_bitset_data_on_another_small() {
        let dataset = BinaryData::read("test_data/small_.txt", false, 0.0);
        let bitset_data = format_data_into_bitset(&dataset);

        let expected_inputs = [[654u64], [214], [108], [197]];
        assert_eq!(bitset_data.inputs.iter().eq(expected_inputs.iter()), true);

        let expected_targets = [[230u64], [793]];
        assert_eq!(bitset_data.targets.iter().eq(expected_targets.iter()), true);

        assert_eq!(bitset_data.chunks, 1);
    }

    #[test]
    fn create_data_structure() {
        let dataset = BinaryData::read("test_data/small_.txt", false, 0.0);
        let structure = Bitset::new(&dataset);

        assert_eq!(structure.support, 10);
        assert_eq!(structure.position.is_empty(), true);
        assert_eq!(structure.num_labels(), 2);
        assert_eq!(structure.num_attributes(), 4);

        if let Some(state) = structure.get_last_state() {
            let expected_state = [1023u64];
            assert_eq!(state.iter().eq(expected_state.iter()), true);
        }
    }

    #[test]
    fn check_backtracking() {
        let dataset = BinaryData::read("test_data/small_.txt", false, 0.0);
        let mut structure = Bitset::new(&dataset);

        structure.push(item(3, 1));
        let s = structure.temp_push(item(2, 1));
        assert_eq!(s, 2);

        structure.push(item(0, 0));
        structure.backtrack();

        let expected_position = [item(3usize, 1usize)];

        assert_eq!(structure.position.iter().eq(expected_position.iter()), true);
        assert_eq!(structure.support, usize::MAX);
        if let Some(state) = structure.get_last_state() {
            let expected_state = [197u64];
            assert_eq!(state.iter().eq(expected_state.iter()), true);
        }
    }

    #[test]
    fn moving_on_step() {
        let dataset = BinaryData::read("test_data/small_.txt", false, 0.0);
        let mut structure = Bitset::new(&dataset);

        let num_attributes = structure.inputs.inputs.len();
        let expected_supports = [5usize, 5, 6, 6];

        for i in 0..num_attributes {
            let support = structure.push(item(i, 0));
            structure.backtrack();
            assert_eq!(support, expected_supports[i]);
        }
    }

    #[test]
    fn check_label_support() {
        let dataset = BinaryData::read("test_data/small_.txt", false, 0.0);
        let mut structure = Bitset::new(&dataset);

        let itemset = [item(0usize, 1usize), item(2, 0), item(3, 0)];
        for elem in itemset.iter() {
            structure.push(*elem);
        }

        assert_eq!(structure.support, 2);
        assert_eq!(structure.label_support(0), 1);
        assert_eq!(structure.label_support(1), 1);
        assert_eq!(structure.labels_support().iter().eq([1, 1].iter()), true);
    }

    #[test]
    fn check_on_large_dataset() {
        let dataset = BinaryData::read("test_data/anneal.txt", false, 0.0);

        let expected_targets = [
            [
                203969538u64,
                395828852097574,
                4820698952720487480,
                3746994894955102272,
                686814682353254433,
                150941104610869889,
                9852328427444797809,
                8816037986354,
                172900954398524432,
                324338632297154705,
                4056909217079535680,
                9223728416196542728,
                18014544890724607,
            ],
            [
                17591982074877,
                18446348244857454041,
                13626045120989064135,
                14699749178754449343,
                17759929391356297182,
                18295802969098681726,
                8594415646264753806,
                18446735257671565261,
                18273843119311027183,
                18122405441412396910,
                14389834856630015935,
                9223015657513008887,
                18428729528818827008,
            ],
        ];
        let expected_inputs = [
            [
                17592186044415u64,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                6783900975316,
                9547810741001883928,
                11562008670787929152,
                2342747057307005028,
                10484771828318722220,
                10115369708795872385,
                2062648630985671777,
                9907919523981689376,
                9971398985973268628,
                2396383689006514501,
                4672841002806551547,
                5180382176735159188,
                793313352578277976,
            ],
            [
                10808279727403,
                6593090318590832355,
                2197886666540235551,
                15527535567609712283,
                5367896590135198019,
                3647560383166790476,
                2469636687664724894,
                8538535343810018519,
                1160935990745792099,
                14897438192833787066,
                13053327130506664960,
                12113282061988502635,
                13856878500531885219,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                4259840,
                541065216,
                2533274790412288,
                576460898332377088,
                2305843009213703168,
                4611686018964783122,
                4611692632693800960,
                289171558105344,
                2702159853731709192,
                1152921504606847488,
                144115188076118016,
                140738058780672,
                301741175570958592,
            ],
            [
                512,
                67108868,
                68719476736,
                704643072,
                68719476736,
                72057594037927968,
                74872343807131648,
                0,
                0,
                67110912,
                576460752303423488,
                0,
                1152921646340767744,
            ],
            [
                1048576,
                0,
                4611686018427387904,
                549755813888,
                2201170739200,
                70368744177664,
                4503599627370496,
                0,
                0,
                0,
                16384,
                1152921504606846976,
                36028797018963968,
            ],
            [
                32768,
                2305843009213693952,
                72066390130950272,
                0,
                288230376151711744,
                0,
                17592320262144,
                34359738376,
                274877972480,
                687195291648,
                16777216,
                17592186044416,
                17592186568704,
            ],
            [
                0,
                4294967296,
                562984313159712,
                256,
                528,
                0,
                9223372586610589696,
                0,
                4612248968380809216,
                0,
                4,
                134217728,
                2305843009482129412,
            ],
            [
                8795952971775,
                18446743523379113983,
                18155980422767427583,
                17870282625621360639,
                5764607517665516287,
                13834776579768057837,
                13546821064864038911,
                17293533397544599295,
                15744584185618104055,
                17293787384730582271,
                18302558516889239551,
                17293681831043923967,
                18108974100045887231,
            ],
            [
                8796233072640,
                550330437632,
                290763650942124032,
                576461448088190976,
                12682136556044035328,
                4611967493941493778,
                4899923008845512704,
                1153210676164952320,
                2702159888091447560,
                1152956688978969344,
                144185556820312064,
                1153062242665627648,
                337769973663664384,
            ],
            [
                8795953496063,
                18446744073134931967,
                18444210798919139327,
                17870282625621360639,
                16140901064495848447,
                13835058054744768493,
                13835051441015750655,
                18446454902151446271,
                15744584219977842423,
                17293822569102671359,
                18302628885633417215,
                17293681831043923967,
                18108974100045887231,
            ],
            [
                8796232548352,
                574619648,
                2533274790412288,
                576461448088190976,
                2305843009213703168,
                4611686018964783122,
                4611692632693800960,
                289171558105344,
                2702159853731709192,
                1152921504606880256,
                144115188076134400,
                1153062242665627648,
                337769973663664384,
            ],
            [
                17592180735999,
                18446744073168486399,
                18444210798919139327,
                17870282625621360639,
                16140901064495848447,
                13835058054744768493,
                13835051441015750655,
                18446454902151446271,
                15744584219977842423,
                17293822569102704127,
                18302628885633417215,
                17293681831043923967,
                18108974101119629055,
            ],
            [
                5308416,
                541065216,
                2533274790412288,
                576461448088190976,
                2305843009213703168,
                4611686018964783122,
                4611692632693800960,
                289171558105344,
                2702159853731709192,
                1152921504606847488,
                144115188076134400,
                1153062242665627648,
                337769972589922560,
            ],
            [
                17592180735999,
                18446744073705357311,
                18446462598732840959,
                17870282625621360639,
                18446744073709542399,
                13835058054744768493,
                18446737459443138559,
                18446737476639784703,
                18338657682652659455,
                17293822569102704127,
                18302628885633417215,
                17293681831580794879,
                18113477701283872767,
            ],
            [
                5308416,
                4194304,
                281474976710656,
                576461448088190976,
                9216,
                4611686018964783122,
                6614266413056,
                6597069766912,
                108086391056892160,
                1152921504606847488,
                144115188076134400,
                1153062242128756736,
                333266372425678848,
            ],
            [
                17592184995839,
                18446744073709551615,
                18446744073709551615,
                17870282771650248703,
                18446744073709543423,
                13835058054744768493,
                18446741874686296063,
                18446739675663040511,
                18338657682652659455,
                18446744073709551615,
                18446744073709535231,
                17293681831614349311,
                18401708077435846655,
            ],
            [
                1048576,
                0,
                0,
                576461302059302912,
                8192,
                4611686018964783122,
                2199023255552,
                4398046511104,
                108086391056892160,
                0,
                16384,
                1153062242095202304,
                45035996273704960,
            ],
            [
                17592184995839,
                18446744073709551615,
                18446744073709551615,
                18446743523953737727,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709535231,
                17293822569102704639,
                18410715276690587647,
            ],
            [
                1048576,
                0,
                0,
                549755813888,
                0,
                0,
                0,
                0,
                0,
                0,
                16384,
                1152921504606846976,
                36028797018963968,
            ],
            [
                15666880696055,
                13690093956180734908,
                17870241251998040047,
                17292555843659234812,
                13688682270762205181,
                18261515845881560247,
                18428553706094853881,
                18285177404584360878,
                18446708888511050750,
                18263691840897076215,
                6629295865958658047,
                16136397460564409311,
                4593671619824447484,
            ],
            [
                1925305348360,
                4756650117528816707,
                576502821711511568,
                1154188230050316803,
                4758061802947346434,
                185228227827991368,
                18190367614697734,
                161566669125190737,
                35185198500865,
                183052232812475400,
                11817448207750893568,
                2310346613145142304,
                13853072453885104131,
            ],
            [
                15942832353015,
                18446458129819369405,
                18446704203458936815,
                17293681829466865148,
                13688682270762729469,
                18298107661573422591,
                18428588924826681081,
                18287429217282949038,
                18446744072883141631,
                18264272417666825207,
                6917526276613767167,
                16140901064487271423,
                13817043656680402940,
            ],
            [
                1649353691400,
                285943890182210,
                39870250614800,
                1153062244242686467,
                4758061802946822146,
                148636412136129024,
                18155148882870534,
                159314856426602577,
                826409984,
                182471656042726408,
                11529217797095784448,
                2305843009222280192,
                4629700417029148675,
            ],
            [
                15942832353279,
                18446458131966853053,
                18446739667003899887,
                17293822569102704126,
                18300368289190117375,
                18298125253759467007,
                18446603323336163067,
                18296436416806125486,
                18446744073688447999,
                18264276815713336319,
                16140898863291465727,
                16140901064495725567,
                13817043656680407036,
            ],
            [
                1649353691136,
                285941742698562,
                4406705651728,
                1152921504606847489,
                146375784519434240,
                148618819950084608,
                140750373388548,
                150307656903426129,
                21103616,
                182467257996215296,
                2305845210418085888,
                2305843009213826048,
                4629700417029144579,
            ],
            [
                17592100057087,
                18446458131966853053,
                18446744073640345583,
                17293822569102704639,
                18444483477265973247,
                18298125258054434303,
                18446603331926228731,
                18297562316712968175,
                18446744073705356287,
                18408393103837757439,
                16140901062314786815,
                16140901064495726591,
                18428729675116183548,
            ],
            [
                85987328,
                285941742698562,
                69206032,
                1152921504606846976,
                2260596443578368,
                148618815655117312,
                140741783322884,
                149181756996583440,
                4195328,
                38350969871794176,
                2305843011394764800,
                2305843009213825024,
                18014398593368067,
            ],
            [
                17592116834303,
                18446462598732840957,
                18446744073642442735,
                17293822569102704639,
                18444483477802844159,
                18298125258054434303,
                18446603331926228735,
                18446744054369615855,
                18446744073705356287,
                18408393108132724735,
                16140901064462270463,
                18446744073709420543,
                18428729675116183548,
            ],
            [
                69210112,
                281474976710658,
                67108880,
                1152921504606846976,
                2260595906707456,
                148618815655117312,
                140741783322880,
                19339935760,
                4195328,
                38350965576826880,
                2305843009247281152,
                131072,
                18014398593368067,
            ],
            [
                17592183943167,
                18446462598732840957,
                18446744073642442735,
                17293822569102704639,
                18444483477802844159,
                18298125275234303487,
                18446603331926228991,
                18446744054369615855,
                18446744073705356287,
                18410644907946409983,
                16140901064462270463,
                18446744073709420543,
                18428729675132960764,
            ],
            [
                2101248,
                281474976710658,
                67108880,
                1152921504606846976,
                2260595906707456,
                148618798475248128,
                140741783322624,
                19339935760,
                4195328,
                36099165763141632,
                2305843009247281152,
                131072,
                18014398576590851,
            ],
            [
                17592186011135,
                16122886665919234035,
                13762991594284253055,
                16140901063791214591,
                18158511427667591167,
                18374616110793228255,
                18367349438443159423,
                18446744039349813239,
                17293822294224732159,
                18446743386447149055,
                17870283317094383615,
                18446726481523507199,
                17293734466438037503,
            ],
            [
                33280,
                2323857407790317580,
                4683752479425298560,
                2305843009918337024,
                288232646041960448,
                72127962916323360,
                79394635266392192,
                34359738376,
                1152921779484819456,
                687262402560,
                576460756615168000,
                17592186044416,
                1153009607271514112,
            ],
            [
                17592186011135,
                16122886665919234035,
                13762991594284253055,
                18446744073004908543,
                18158511427667591167,
                18374616110927445983,
                18367349438443159423,
                18446744039349813239,
                17293822294224732159,
                18446743386447149055,
                17870283317094383615,
                18446726481523507199,
                17293734466438037503,
            ],
            [
                33280,
                2323857407790317580,
                4683752479425298560,
                704643072,
                288232646041960448,
                72127962782105632,
                79394635266392192,
                34359738376,
                1152921779484819456,
                687262402560,
                576460756615168000,
                17592186044416,
                1153009607271514112,
            ],
            [
                17592186043903,
                18428729675132927987,
                13835057984415203327,
                18446744073004908543,
                18446741803819302911,
                18374616110927445983,
                18367367030763421567,
                18446744073709551615,
                17293822569102704639,
                18446744073642440703,
                17870283317111160831,
                18446744073709551615,
                17293752058624606207,
            ],
            [
                512,
                18014398576623628,
                4611686089294348288,
                704643072,
                2269890248704,
                72127962782105632,
                79377042946130048,
                0,
                1152921504606846976,
                67110912,
                576460756598390784,
                0,
                1152992015084945408,
            ],
            [
                17592186044415,
                18428729675132927987,
                13835058053134680063,
                18446744073038462975,
                18446741803819302911,
                18446673704965373951,
                18439424624803446655,
                18446744073709551615,
                17293822569102704639,
                18446744073642440703,
                17870283317111160831,
                18446744073709551615,
                18446673567526420479,
            ],
            [
                0,
                18014398576623628,
                4611686020574871552,
                671088640,
                2269890248704,
                70368744177664,
                7319448906104960,
                0,
                1152921504606846976,
                67110912,
                576460756598390784,
                0,
                70506183131136,
            ],
            [
                17592186044415,
                18428729675132927987,
                18446744071562067967,
                18446744073038462975,
                18446744004990042111,
                18446744073709551615,
                18439424624803446655,
                18446744073709551615,
                17293822569102704639,
                18446744073642440703,
                17870283317111160831,
                18446744073709551615,
                18446673567526420479,
            ],
            [
                0,
                18014398576623628,
                2147483648,
                671088640,
                68719509504,
                0,
                7319448906104960,
                0,
                1152921504606846976,
                67110912,
                576460756598390784,
                0,
                70506183131136,
            ],
            [
                17592186044415,
                18446744073642442747,
                18446744073709551615,
                18446744073038462975,
                18446744004990074879,
                18446744073709551615,
                18439425724315074559,
                18446744073709551615,
                18446744073709551615,
                18446744073642440703,
                17870283321406128127,
                18446744073709551615,
                18446743936270598143,
            ],
            [
                0,
                67108868,
                0,
                671088640,
                68719476736,
                0,
                7318349394477056,
                0,
                0,
                67110912,
                576460752303423488,
                0,
                137438953472,
            ],
            [
                17592186044415,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18442240474082181119,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
            ],
            [0, 0, 0, 0, 0, 0, 4503599627370496, 0, 0, 0, 0, 0, 0],
            [
                14681430923178,
                2396512812592796742,
                14941665774570759384,
                13385481196665005993,
                7522640860962665922,
                14503115606916189938,
                13931761220469858438,
                1963515974876171035,
                8470161711737101580,
                8617713906285057552,
                8226086686264459484,
                6928137400968119873,
                15380994559860689667,
            ],
            [
                2910755121237,
                16050231261116754873,
                3505078299138792231,
                5061262877044545622,
                10924103212746885693,
                3943628466793361677,
                4514982853239693177,
                16483228098833380580,
                9976582361972450035,
                9829030167424494063,
                10220657387445092131,
                11518606672741431742,
                3065749513848861948,
            ],
            [
                4312268800,
                144696279973237249,
                9223373136366403584,
                28187080090845184,
                2533274861699072,
                186336726649669640,
                18119951894448130,
                1152921574400099456,
                5104467969,
                9262497058751787008,
                4688357773241487872,
                10133210835124224,
                792668718923534338,
            ],
            [
                17587873775615,
                18302047793736314366,
                9223370937343148031,
                18418556993618706431,
                18444210798847852543,
                18260407347059881975,
                18428624121815103485,
                17293822499309452159,
                18446744068605083646,
                9184247014957764607,
                13758386300468063743,
                18436610862874427391,
                17654075354786017277,
            ],
            [
                4857545093,
                9369203433734935073,
                9223412724187353088,
                14012006861735378946,
                7515241158071549955,
                2492179735863371786,
                5062151534549407762,
                3770075908745059456,
                4522298579361803,
                9343566318860792088,
                13911738881067260672,
                6999725330072272896,
                5404363808355405826,
            ],
            [
                17587328499322,
                9077540639974616542,
                9223331349522198527,
                4434737211974172669,
                10931502915638001660,
                15954564337846179829,
                13384592539160143853,
                14676668164964492159,
                18442221775130189812,
                9103177754848759527,
                4535005192642290943,
                11447018743637278719,
                13042380265354145789,
            ],
            [
                176656245141,
                10522124938341790257,
                12105718702042861572,
                14014840303479603586,
                7515312076571545735,
                3645101241543960650,
                5062160436506662938,
                12993660013905828992,
                9227929519807308299,
                11653915127004066078,
                14236578596915321600,
                7035754419149307910,
                5404367106891345922,
            ],
            [
                17415529799274,
                7924619135367761358,
                6341025371666690043,
                4431903770229948029,
                10931431997138005880,
                14801642832165590965,
                13384583637202888677,
                5453084059803722623,
                9218814553902243316,
                6792828946705485537,
                4210165476794230015,
                11410989654560243705,
                13042376966818205693,
            ],
            [
                11171772656031,
                15818359216846158769,
                13258640206922338373,
                14014840372203282839,
                7528840476263914199,
                3654262372495800650,
                5062442048922393626,
                12995911826740739201,
                13839765210395880047,
                11817170613496246558,
                14245585796314766243,
                7331338878823401606,
                5404367742546702338,
            ],
            [
                6420413388384,
                2628384856863392846,
                5188103866787213242,
                4431903701506268776,
                10917903597445637416,
                14792481701213750965,
                13384302024787157989,
                5450832246968812414,
                4606978863313671568,
                6629573460213305057,
                4201158277394785372,
                11115405194886150009,
                13042376331162849277,
            ],
            [
                11738779969951,
                15818680276658486265,
                18158556678827923557,
                16320692594256045567,
                18085284878275533783,
                3654614216367847882,
                16670472296218304059,
                13140049022229020389,
                13923651934657899135,
                12393675487999424799,
                16569445410578176943,
                7331620353951125126,
                5407059484788092930,
            ],
            [
                5853406074464,
                2628063797051065350,
                288187394881628058,
                2126051479453506048,
                361459195434017832,
                14792129857341703733,
                1776271777491247556,
                5306695051480531226,
                4523092139051652480,
                6053068585710126816,
                1877298663131374672,
                11115123719758426489,
                13039684588921458685,
            ],
            [
                11750593227679,
                16113736425328278521,
                18257354395653495655,
                17185383723298385407,
                18085286050803973111,
                3942844601646627806,
                17870264353739333179,
                13140095201731149543,
                16229495304817143551,
                12393675765630105983,
                16569727022994004975,
                8070360363869658790,
                5408185387379850294,
            ],
            [
                5841592816736,
                2333007648381273094,
                189389678056055960,
                1261360350411166208,
                361458022905578504,
                14503899472062923809,
                576479719970218436,
                5306648871978402072,
                2217248768892408064,
                6053068308079445632,
                1877017050715546640,
                10376383709839892825,
                13038558686329701321,
            ],
            [
                12025471168511,
                16131891631321251837,
                18446664359038410607,
                17257512287376175103,
                18157343644850323447,
                4035187291868019711,
                18446743798830981115,
                13248463067764752103,
                17546024139645447935,
                12681923795175340543,
                18302628885633695743,
                17293802775911595966,
                15789620258114895102,
            ],
            [
                5566714875904,
                2314852442388299778,
                79714671141008,
                1189231786333376512,
                289400428859228168,
                14411556781841531904,
                274878570500,
                5198281005944799512,
                900719934064103680,
                5764820278534211072,
                144115188075855872,
                1152941297797955649,
                2657123815594656513,
            ],
            [
                618477650082,
                144211945099104320,
                13835063552857114120,
                306244774661210120,
                5350311681274939648,
                3458834938399293440,
                9233786757571420162,
                3603020442069631568,
                8933549932800,
                4758054390403792896,
                2305843078202267659,
                285873300046889,
                1152921504615235590,
            ],
            [
                16973708394333,
                18302532128610447295,
                4611680520852437495,
                18140499299048341495,
                13096432392434611967,
                14987909135310258175,
                9212957316138131453,
                14843723631639920047,
                18446735140159618815,
                13688689683305758719,
                16140900995507283956,
                18446458200409504726,
                17293822569094316025,
            ],
            [
                1719198421218,
                756842508937334977,
                13835064119804397080,
                452617260410881289,
                5350313897614477568,
                3530896656679587876,
                9382405549603184642,
                3928449775882014299,
                7910027366143810,
                5086826165457502226,
                2350888147063806287,
                938169310051043049,
                1297037580979675302,
            ],
            [
                15872987623197,
                17689901564772216638,
                4611679953905154535,
                17994126813298670326,
                13096430176095074047,
                14915847417029963739,
                9064338524106366973,
                14518294297827537316,
                18438834046343407805,
                13359917908252049389,
                16095855926645745328,
                17508574763658508566,
                17149706492729876313,
            ],
            [
                15336472498415,
                9987820998516276435,
                16206341270648116829,
                468116270682269081,
                15764678038637759427,
                13621079820101252348,
                10355416167998186582,
                3959980067205545695,
                10627716279143735242,
                5763971398696697047,
                8136031852222421343,
                4572769763632049897,
                1353288600757248183,
            ],
            [
                2255713546000,
                8458923075193275180,
                2240402803061434786,
                17978627803027282534,
                2682066035071792188,
                4825664253608299267,
                8091327905711365033,
                14486764006504005920,
                7819027794565816373,
                12682772675012854568,
                10310712221487130272,
                13873974310077501718,
                17093455472952303432,
            ],
            [
                15336472563951,
                9987820999124450519,
                16206622891721086557,
                4503351029220894137,
                18070521116570939331,
                18232765839468721406,
                14969919153486324822,
                4536725593020563423,
                13257818538837516234,
                6916963289294700247,
                8280147040298539359,
                4572769764219252713,
                1655029913230289343,
            ],
            [
                2255713480464,
                8458923074585101096,
                2240121181988465058,
                13943393044488657478,
                376222957138612284,
                213978234240830209,
                3476824920223226793,
                13910018480688988192,
                5188925534872035381,
                11529780784414851368,
                10166597033411012256,
                13873974309490298902,
                16791714160479262272,
            ],
            [
                15337026213103,
                9988947517506584319,
                16643472330487488349,
                9115179984164087739,
                18077443646074384323,
                18437697223291107070,
                17275762197126866006,
                4536725610477256671,
                13257888908118564814,
                9223369248478592735,
                17505207927114275295,
                13798393600887717865,
                2233746863393978815,
            ],
            [
                2255159831312,
                8457796556202967296,
                1803271743222063266,
                9331564089545463876,
                369300427635167292,
                9046850418444545,
                1170981876582685609,
                13910018463232294944,
                5188855165590986801,
                9223374825230958880,
                941536146595276320,
                4648350472821833750,
                16212997210315572800,
            ],
            [
                17536049468655,
                12592029201638359039,
                17869577329039106015,
                9115179988526167995,
                18365674056586358723,
                18446704491265324798,
                17275762197263246454,
                4536725644972261375,
                13257889183130689486,
                9223369798234930911,
                17505207927131053535,
                13798958749931503593,
                2233746863393979839,
            ],
            [
                56136575760,
                5854714872071192576,
                577166744670445600,
                9331564085183383620,
                81070017123192892,
                39582444226817,
                1170981876446305161,
                13910018428737290240,
                5188854890578862129,
                9223374275474620704,
                941536146578498080,
                4647785323778048022,
                16212997210315571776,
            ],
            [
                17592152489983,
                18446744073692774399,
                18446040351908036607,
                18446744073709551615,
                18374686479670575103,
                18446739675662778366,
                18446744073709518718,
                18446744073709551615,
                18446744073709551615,
                18446744073709551615,
                18446462598732832767,
                18446673704964325375,
                18446744004990074815,
            ],
            [
                33554432,
                16777216,
                703721801515008,
                0,
                72057594038976512,
                4398046773249,
                32897,
                0,
                0,
                0,
                281474976718848,
                70368745226240,
                68719476800,
            ],
            [
                14681430923178,
                2396512812592796742,
                14941665774571807960,
                13385481196665005993,
                7522640860962665922,
                14503115607990980338,
                13931761220469858438,
                1963515974876171035,
                8470161711737101580,
                8617713906285073936,
                8226086686264459484,
                6928137400968119873,
                15380994559894244099,
            ],
            [
                2910755121237,
                16050231261116754873,
                3505078299137743655,
                5061262877044545622,
                10924103212746885693,
                3943628465718571277,
                4514982853239693177,
                16483228098833380580,
                9976582361972450035,
                9829030167424477679,
                10220657387445092131,
                11518606672741431742,
                3065749513815307516,
            ],
            [
                15233334220715,
                2396583460509914182,
                14941806512060228856,
                13673711572817504169,
                7666756049040618946,
                14539144413600145143,
                14517229322351911318,
                1999579964932655899,
                17711618519070997772,
                8617995415621989904,
                8226086763574012126,
                7000199410232428099,
                15525109750117591843,
            ],
            [
                2358851823700,
                16050160613199637433,
                3504937561649322759,
                4773032500892047446,
                10779988024668932669,
                3907599660109406472,
                3929514751357640297,
                16447164108776895716,
                735125554638553843,
                9828748658087561711,
                10220657310135539489,
                11446544663477123516,
                2921634323591959772,
            ],
            [
                15233879480235,
                2396583460510446694,
                14941808711083484408,
                13673711572835329961,
                7666756598796432835,
                14541396213413830391,
                14517264541083738518,
                11222952001787431707,
                17711618519070997788,
                8627565564838540816,
                8519946673619675358,
                7000199410232428099,
                15525111949140847395,
            ],
            [
                2358306564180,
                16050160613199104921,
                3504935362626067207,
                4773032500874221654,
                10779987474913118780,
                3905347860295721224,
                3929479532625813097,
                7223792071922119908,
                735125554638553827,
                9819178508871010799,
                9926797400089876257,
                11446544663477123516,
                2921632124568704220,
            ],
            [
                17450384603055,
                3276368693182692590,
                14942934628170198264,
                18286857765320130559,
                9110745366796287447,
                14983316324917043191,
                14949698686899169214,
                13528795646790512479,
                18302030751215910719,
                18445565309900289555,
                8555977274862563039,
                16658168845756784195,
                17867713900087049015,
            ],
            [
                141801441360,
                15170375380526859025,
                3503809445539353351,
                159886308389421056,
                9335998706913264168,
                3463427748792508424,
                3497045386810382401,
                4917948426919039136,
                144713322493640896,
                1178763809262060,
                9890766798846988576,
                1788575227952767420,
                579030173622502600,
            ],
            [
                17450451842991,
                3276368693182692590,
                14951941965401288440,
                18286857765454348287,
                9182804129066368471,
                14983316462355996663,
                14951950487790807039,
                13546810050134060895,
                18446145939291766719,
                18446691227054112339,
                8556258749839273695,
                17253286111361368043,
                18444175339602017279,
            ],
            [
                141734201424,
                15170375380526859025,
                3494802108308263175,
                159886308255203328,
                9263939944643183144,
                3463427611353554952,
                3494793585918744576,
                4899934023575490720,
                598134417784896,
                52846655439276,
                9890485323870277920,
                1193457962348183572,
                2568734107534336,
            ],
            [
                17592186011647,
                16140894466889220095,
                18446673704898264959,
                17293822560512703999,
                18446744073172680703,
                13258597300528742399,
                18446744073708896251,
                17869720371452706559,
                18410715268100653055,
                17293752195526687231,
                18302626686610440191,
                18446726480953081855,
                18149506498303098622,
            ],
            [
                32768,
                2305843009213693952,
                0,
                8590000128,
                0,
                2148007936,
                0,
                256,
                36028797018963968,
                536870912,
                0,
                0,
                257,
            ],
            [
                0,
                6597606637568,
                70368811286656,
                1152921504606847488,
                536870912,
                5188146771032801280,
                655364,
                577023702256844800,
                8589934592,
                1152991877645993472,
                144117387099111424,
                17592756469760,
                297237575406452736,
            ],
        ];

        let structure = Bitset::new(&dataset);
        let expected_chunks = 13usize;

        assert_eq!(structure.inputs.chunks, expected_chunks);
        assert_eq!(
            structure.inputs.inputs.iter().eq(expected_inputs.iter()),
            true
        );
        assert_eq!(
            structure.inputs.targets.iter().eq(expected_targets.iter()),
            true
        );
    }

    #[test]
    fn check_reset() {
        let dataset = BinaryData::read("test_data/anneal.txt", false, 0.0);
        let mut structure = Bitset::new(&dataset);

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
    fn see_tids() {
        let dataset = BinaryData::read("test_data/rsparse_dataset.txt", false, 0.0);
        let mut structure = Bitset::new(&dataset);
        // Print in binary

        let sup = structure.push(item(0, 0));
        println!("usize: {}", sup);
        let state = structure.get_last_state();
        if let Some(state) = state {
            for chunk in state.iter() {
                println!("{:064b}", chunk);
            }
        }

        println!("Tids: {:?}", structure.get_tids());
    }
}
