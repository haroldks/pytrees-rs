mod rules;

use crate::globals::{compute_entropy, item};
use crate::searches::errors::classification_error;
use crate::structures::Structure;
use float_cmp::{ApproxEq, F64Margin};

pub trait Heuristic {
    fn compute(&self, structure: &mut dyn Structure, candidates: &mut Vec<usize>);
}

#[derive(Default)]
pub struct NoHeuristic;

impl Heuristic for NoHeuristic {
    fn compute(&self, _structure: &mut dyn Structure, _candidates: &mut Vec<usize>) {}
}

#[derive(Default)]
pub struct GiniIndex;

impl Heuristic for GiniIndex {
    fn compute(&self, structure: &mut dyn Structure, candidates: &mut Vec<usize>) {
        let root_classes_support = structure.labels_support().to_vec();
        let mut candidates_sorted = vec![];
        for attribute in candidates.iter() {
            let gini = Self::gini_index(*attribute, structure, &root_classes_support);
            candidates_sorted.push((*attribute, gini));
        }
        candidates_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        *candidates = candidates_sorted
            .iter()
            .map(|(a, _)| *a)
            .collect::<Vec<usize>>();
    }
}

impl GiniIndex {
    fn gini_index(
        attribute: usize,
        structure: &mut dyn Structure,
        root_classes_support: &[usize],
    ) -> f64 {
        let _ = structure.push(item(attribute, 0));
        let left_classes_supports = structure.labels_support().to_vec();
        structure.backtrack();

        let right_classes_support = root_classes_support
            .iter()
            .enumerate()
            .map(|(idx, val)| *val - left_classes_supports[idx])
            .collect::<Vec<usize>>();

        let actual_size = root_classes_support.iter().sum::<usize>() as f64;
        let left_split_size = left_classes_supports.iter().sum::<usize>();
        let right_split_size = right_classes_support.iter().sum::<usize>();

        let mut left_gini_index = 0f64;
        let mut right_gini_index = 0f64;

        for class in 0..root_classes_support.len() {
            let p = match left_split_size {
                0 => 0f64,
                _ => (left_classes_supports[class] as f64 / left_split_size as f64).powf(2.),
            };

            left_gini_index += p;

            let p = match right_split_size {
                0 => 0f64,
                _ => (right_classes_support[class] as f64 / right_split_size as f64).powf(2.),
            };

            right_gini_index += p
        }
        ((left_split_size as f64) * (1. - left_gini_index)
            + (right_split_size as f64) * (1. - right_gini_index))
            / actual_size
    }
}

#[derive(Default)]
pub struct InformationGain;

impl Handler for InformationGain {}

impl Heuristic for InformationGain {
    fn compute(&self, structure: &mut dyn Structure, candidates: &mut Vec<usize>) {
        self.internally_compute(structure, candidates, false);
    }
}

#[derive(Default)]
pub struct InformationGainRatio;

impl Handler for InformationGainRatio {}

impl Heuristic for InformationGainRatio {
    fn compute(&self, structure: &mut dyn Structure, candidates: &mut Vec<usize>) {
        self.internally_compute(structure, candidates, true);
    }
}

// Information Gain and Information Gain Ratio handler

trait Handler {
    fn internally_compute(
        &self,
        structure: &mut dyn Structure,
        attributes: &mut Vec<usize>,
        ratio: bool,
    ) {
        let root_classes_support = structure.labels_support().to_vec();
        let parent_entropy = compute_entropy(&root_classes_support);
        let mut candidates_sorted = vec![];
        for attribute in attributes.iter() {
            let info_gain = Self::information_gain(
                *attribute,
                structure,
                &root_classes_support,
                parent_entropy,
                ratio,
            );
            candidates_sorted.push((*attribute, info_gain));
        }
        candidates_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        *attributes = candidates_sorted
            .iter()
            .map(|(a, _)| *a)
            .collect::<Vec<usize>>();
    }

    fn information_gain(
        attribute: usize,
        structure: &mut dyn Structure,
        root_classes_support: &[usize],
        parent_entropy: f64,
        ratio: bool,
    ) -> f64 {
        let _ = structure.push(item(attribute, 0));
        let left_classes_supports = structure.labels_support().to_vec();
        structure.backtrack();

        let right_classes_support = root_classes_support
            .iter()
            .enumerate()
            .map(|(idx, val)| *val - left_classes_supports[idx])
            .collect::<Vec<usize>>();

        let actual_size = root_classes_support.iter().sum::<usize>();
        let left_split_size = left_classes_supports.iter().sum::<usize>();
        let right_split_size = right_classes_support.iter().sum::<usize>();

        let left_weight = match actual_size {
            0 => 0f64,
            _ => left_split_size as f64 / actual_size as f64,
        };

        let right_weight = match actual_size {
            0 => 0f64,
            _ => right_split_size as f64 / actual_size as f64,
        };

        let mut split_info = 0f64;
        if ratio {
            if left_weight > 0. {
                split_info = -left_weight * left_weight.log2();
            }
            if right_weight > 0. {
                split_info += -right_weight * right_weight.log2();
            }
        }
        if split_info.approx_eq(
            0.,
            F64Margin {
                ulps: 2,
                epsilon: 0.0,
            },
        ) {
            split_info = 1f64;
        }

        let left_split_entropy = compute_entropy(&left_classes_supports);
        let right_split_entropy = compute_entropy(&right_classes_support);

        let info_gain = parent_entropy
            - (left_weight * left_split_entropy + right_weight * right_split_entropy);
        if ratio {
            return info_gain / split_info;
        }
        info_gain
    }
}

#[derive(Default)]
pub struct Purity;

impl Heuristic for Purity {
    fn compute(&self, structure: &mut dyn Structure, candidates: &mut Vec<usize>) {
        let root_classes_support = structure.labels_support().to_vec();
        let mut candidates_sorted = vec![];
        for attribute in candidates.iter() {
            let purity = Self::purity_by_attribute(*attribute, structure, &root_classes_support);
            candidates_sorted.push((*attribute, purity));
        }
        candidates_sorted.sort_by(|a, b| 0.1.partial_cmp(&a.1).unwrap());
        *candidates = candidates_sorted
            .iter()
            .map(|(a, _)| *a)
            .collect::<Vec<usize>>();
    }
}

impl Purity {
    fn purity_by_attribute(
        attribute: usize,
        structure: &mut dyn Structure,
        root_classes_support: &[usize],
    ) -> f64 {
        let _ = structure.push(item(attribute, 0));
        let left_classes_supports = structure.labels_support().to_vec();

        let left_error = classification_error(&left_classes_supports);

        structure.backtrack();

        let total = root_classes_support.iter().sum::<usize>() as f64;

        let right_classes_support = root_classes_support
            .iter()
            .enumerate()
            .map(|(idx, val)| *val - left_classes_supports[idx])
            .collect::<Vec<usize>>();

        let right_error = classification_error(&right_classes_support);

        (left_error.0 + right_error.0) / total
    }
}
