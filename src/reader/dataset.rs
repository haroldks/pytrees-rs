use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use super::{DataFormat, DataReaderError};

/// Continuous dataset: column-major features and integer class labels.
///
/// `features[j]` holds the values of feature `j` for all samples.
/// `labels[i]` is the class index (0-based) of sample `i`.
pub struct Dataset {
    pub features: Vec<Vec<f64>>,
    pub labels: Vec<usize>,
}

impl Dataset {
    pub fn new(features: Vec<Vec<f64>>, labels: Vec<usize>) -> Self {
        Self { features, labels }
    }

    pub fn n_samples(&self) -> usize {
        self.labels.len()
    }

    pub fn n_features(&self) -> usize {
        self.features.len()
    }
}

/// Reads continuous-valued CSV/space/TSV files into a `Dataset`.
///
/// Mirrors the builder API of `DataReader` so both readers feel consistent.
pub struct ContinuousDataReader {
    format: DataFormat,
    has_headers: bool,
    comment_char: Option<char>,
    label_column: Option<usize>,
}

impl Default for ContinuousDataReader {
    fn default() -> Self {
        Self {
            format: DataFormat::Space,
            has_headers: false,
            comment_char: Some('#'),
            label_column: Some(0),
        }
    }
}

impl ContinuousDataReader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_format(mut self, format: DataFormat) -> Self {
        self.format = format;
        self
    }

    pub fn with_headers(mut self, has_headers: bool) -> Self {
        self.has_headers = has_headers;
        self
    }

    pub fn with_comment_char(mut self, comment_char: Option<char>) -> Self {
        self.comment_char = comment_char;
        self
    }

    pub fn with_label_column(mut self, label_column: Option<usize>) -> Self {
        self.label_column = label_column;
        self
    }

    pub fn auto_detect_format(mut self, path: &Path) -> Self {
        self.format = DataFormat::from_extension(path);
        self
    }

    /// Read a file into a `Dataset`.
    ///
    /// Each row is one sample. The label column (default: column 0) must contain
    /// an integer class index. All other columns are parsed as `f64` features.
    pub fn read_file(&self, path: &Path) -> Result<Dataset, DataReaderError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let delimiter = self.format.delimiter();
        let mut row_idx = 0;

        let mut features: Vec<Vec<f64>> = vec![];
        let mut labels: Vec<usize> = vec![];

        for (i, line_result) in reader.lines().enumerate() {
            let line = line_result?;

            if line.trim().is_empty() {
                continue;
            }

            if let Some(comment) = self.comment_char {
                if line.trim().starts_with(comment) {
                    continue;
                }
            }

            if i == 0 && self.has_headers {
                continue;
            }

            let tokens: Vec<&str> = line.split(delimiter).map(|s| s.trim()).collect();

            // On the first data row, allocate feature columns.
            if row_idx == 0 {
                let n_feature_cols = if self.label_column.is_some() {
                    tokens.len() - 1
                } else {
                    tokens.len()
                };
                features = vec![Vec::new(); n_feature_cols];
            }

            let mut feat_idx = 0;
            for (col_idx, &token) in tokens.iter().enumerate() {
                if Some(col_idx) == self.label_column {
                    let label = token.parse::<usize>().map_err(|_| {
                        DataReaderError::Parse(format!(
                            "Expected integer label at line {}, column {}; got '{}'",
                            i + 1,
                            col_idx + 1,
                            token
                        ))
                    })?;
                    labels.push(label);
                } else {
                    let value = token.parse::<f64>().map_err(|_| {
                        DataReaderError::Parse(format!(
                            "Expected float at line {}, column {}; got '{}'",
                            i + 1,
                            col_idx + 1,
                            token
                        ))
                    })?;
                    if feat_idx < features.len() {
                        features[feat_idx].push(value);
                    }
                    feat_idx += 1;
                }
            }

            row_idx += 1;
        }

        Ok(Dataset { features, labels })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_accessors() {
        let features = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let labels = vec![0, 1, 0];
        let ds = Dataset::new(features, labels);
        assert_eq!(ds.n_samples(), 3);
        assert_eq!(ds.n_features(), 2);
    }
}
