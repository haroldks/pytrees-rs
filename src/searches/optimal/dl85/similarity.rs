use crate::structures::{DataCover, Structure};

// Contains the cover of the current data in form of Vec<usize>. To compute the similarity
#[derive(Default)]
pub struct SimilarityCover {
    pub first: Option<DataCover>,
    pub second: Option<DataCover>,
}

impl SimilarityCover {
    pub fn update<S: Structure>(&mut self, error: f64, structure: &mut S) {
        // Check if a Data cover is set otherwise compute it
        let mut data_cover = structure.get_data_cover();
        data_cover.error = error;
        if self.first.is_none() {
            self.first = Some(data_cover);
            return;
        } else if self.second.is_none() {
            self.second = Some(data_cover);
            return;
        }

        let (mut first_in, mut first_out) = (0, 0);
        if let Some(first) = &self.first {
            (first_in, first_out) = structure.get_difference(first)
        }

        let (mut second_in, mut second_out) = (0, 0);
        if let Some(second) = &self.second {
            (second_in, second_out) = structure.get_difference(second)
        }

        if first_in + first_out < second_in + second_out {
            if let Some(ref mut first) = self.first {
                first.update(data_cover)
            }
        } else if let Some(ref mut second) = self.second {
            second.update(data_cover)
        }
    }

    pub fn compute_similarity<S: Structure>(&mut self, structure: &mut S) -> f64 {
        let mut bound = 0.0;
        let saved = [&self.first, &self.second];
        for similarity in saved {
            if similarity.is_some() {
                if let Some(data_cover) = similarity {
                    let (_, diff) = structure.get_difference(data_cover);
                    bound = <f64>::max(bound, data_cover.error - diff as f64);
                }
            }
        }
        bound
    }
}
