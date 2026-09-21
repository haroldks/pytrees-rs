use super::{DataFormat, DataReaderError};
use crate::data::{DataPoint, Dataset};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Reads a delimited text file into a [`Dataset`].
///
/// The default matches the format the datasets in this repository use: one
/// instance per line, whitespace separated, **the label in column 0**, `#`
/// starting a comment, no header row.
///
/// Labels must be non-negative integers. They are treated as a dense encoding,
/// so `num_labels` is `max_label + 1` and a file whose labels are `{0, 2}`
/// simply has an empty class 1.
pub struct DataReader {
    format: DataFormat,
    has_headers: bool,
    comment_char: Option<char>,
    label_column: usize,
}

impl Default for DataReader {
    fn default() -> Self {
        Self {
            format: DataFormat::Space,
            has_headers: false,
            comment_char: Some('#'),
            label_column: 0,
        }
    }
}

impl DataReader {
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

    /// Which column holds the label. Defaults to 0.
    pub fn with_label_column(mut self, label_column: usize) -> Self {
        self.label_column = label_column;
        self
    }

    pub fn auto_detect_format(mut self, path: &Path) -> Self {
        self.format = DataFormat::from_extension(path);
        self
    }

    pub fn read_file(&self, path: &Path) -> Result<Dataset, DataReaderError> {
        let file = File::open(path)?;
        self.read(BufReader::new(file))
    }

    /// Reads from any line source. `read_file` is this over a file.
    pub fn read<R: BufRead>(&self, source: R) -> Result<Dataset, DataReaderError> {
        let delimiter = self.format.delimiter();

        let mut dataset = Dataset::new();
        let mut row_idx = 0usize;
        let mut max_label = 0usize;
        // Fixed by the first data row; every later row must agree, otherwise
        // the columns silently shift and the dataset is quietly wrong.
        let mut num_columns: Option<usize> = None;
        let mut header_pending = self.has_headers;

        for (line_idx, line_result) in source.lines().enumerate() {
            let line = line_result?;
            let line = line.trim();
            let line_no = line_idx + 1;

            if line.is_empty() {
                continue;
            }
            if let Some(comment) = self.comment_char {
                if line.starts_with(comment) {
                    continue;
                }
            }
            // The header is the first *data* line, not line 0: a file may open
            // with comments or blank lines.
            if header_pending {
                header_pending = false;
                continue;
            }

            let tokens: Vec<&str> = if delimiter == ' ' {
                line.split_whitespace().collect()
            } else {
                line.split(delimiter).map(str::trim).collect()
            };

            match num_columns {
                None => {
                    if tokens.len() < 2 {
                        return Err(DataReaderError::Format(format!(
                            "line {line_no}: expected a label and at least one feature, found {} \
                             column(s)",
                            tokens.len()
                        )));
                    }
                    if self.label_column >= tokens.len() {
                        return Err(DataReaderError::Format(format!(
                            "line {line_no}: label column {} is out of range, the file has {} \
                             columns",
                            self.label_column,
                            tokens.len()
                        )));
                    }
                    num_columns = Some(tokens.len());
                }
                Some(expected) if tokens.len() != expected => {
                    return Err(DataReaderError::Format(format!(
                        "line {line_no}: {} columns, expected {expected}",
                        tokens.len()
                    )));
                }
                Some(_) => {}
            }

            let label_token = tokens[self.label_column];
            let label: usize = label_token.parse().map_err(|_| {
                DataReaderError::Parse(format!(
                    "line {line_no}, column {}: `{label_token}` is not a label; labels must be \
                     non-negative integers",
                    self.label_column + 1
                ))
            })?;
            max_label = max_label.max(label);

            for (col_idx, &token) in tokens.iter().enumerate() {
                if col_idx == self.label_column {
                    continue;
                }

                let value = token.parse::<f64>().map_err(|_| {
                    DataReaderError::Parse(format!(
                        "line {line_no}, column {}: `{token}` is not a number",
                        col_idx + 1
                    ))
                })?;
                if !value.is_finite() {
                    // NaN and the infinities survive `parse::<f64>` and then
                    // poison every comparison the search makes on this column.
                    return Err(DataReaderError::Parse(format!(
                        "line {line_no}, column {}: `{token}` is not a finite number",
                        col_idx + 1
                    )));
                }

                let feature_idx = col_idx - usize::from(col_idx > self.label_column);
                dataset.insert(DataPoint::new(row_idx, value, label as f64), feature_idx);
            }
            row_idx += 1;
        }

        if row_idx == 0 {
            return Err(DataReaderError::Format(
                "no data rows: the file is empty or entirely comments".to_string(),
            ));
        }

        // The label histogram is indexed by label, so `num_labels` has to be
        // `max + 1` and not the number of distinct labels -- with labels {0, 5}
        // the latter is 2, and the first lookup of class 5 goes out of bounds.
        // A label id that exceeds the row count cannot be a dense encoding, and
        // would otherwise ask for an absurd allocation.
        if max_label >= row_idx {
            return Err(DataReaderError::Format(format!(
                "label {max_label} in a file of {row_idx} rows: labels must be a dense encoding \
                 starting at 0"
            )));
        }
        dataset.set_num_label(max_label + 1);
        // Both preparation steps happen here rather than being left to the
        // caller: skipping the second one makes every fit return a single leaf,
        // with nothing to say why.
        dataset.sort_features();
        dataset.compute_unique_feature_values();
        Ok(dataset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::DataFormat;
    use crate::tests::fixture;

    fn read(text: &str) -> Result<Dataset, DataReaderError> {
        DataReader::default().read(text.as_bytes())
    }

    fn read_csv(text: &str) -> Result<Dataset, DataReaderError> {
        DataReader::default()
            .with_format(DataFormat::Csv)
            .read(text.as_bytes())
    }

    #[test]
    fn reads_the_label_from_column_zero() {
        let dataset = read("0 1.0 2.0\n1 3.0 4.0\n").unwrap();
        assert_eq!(dataset.count(), 2);
        assert_eq!(dataset.num_features(), 2);
        assert_eq!(dataset.num_labels(), 2);
        assert_eq!(dataset[0][0].value(), 1.0);
        assert_eq!(dataset[1][1].value(), 4.0);
        assert_eq!(dataset[0][1].label(), 1.0);
    }

    #[test]
    fn num_labels_is_max_plus_one_not_the_distinct_count() {
        // Labels {0, 5} used to report num_labels == 2, and the first lookup
        // of class 5 in a length-2 histogram went out of bounds.
        let dataset = read("0 1.0\n5 2.0\n0 3.0\n5 4.0\n5 5.0\n0 6.0\n").unwrap();
        assert_eq!(dataset.num_labels(), 6);
    }

    #[test]
    fn a_label_that_cannot_be_a_dense_encoding_is_rejected() {
        let err = read("0 1.0\n900 2.0\n").unwrap_err();
        assert!(matches!(err, DataReaderError::Format(_)), "{err}");
    }

    #[test]
    fn ragged_rows_are_an_error_not_a_silent_shift() {
        let wider = read("0 1.0 2.0\n1 3.0 4.0 5.0\n").unwrap_err();
        assert!(matches!(wider, DataReaderError::Format(_)), "{wider}");

        let narrower = read("0 1.0 2.0\n1 3.0\n").unwrap_err();
        assert!(matches!(narrower, DataReaderError::Format(_)), "{narrower}");
    }

    #[test]
    fn a_missing_csv_field_is_reported_rather_than_deleted() {
        // `1,,3` used to yield two tokens, shifting every later column left.
        let err = read_csv("0,1.0,2.0\n1,,3.0\n").unwrap_err();
        assert!(matches!(err, DataReaderError::Parse(_)), "{err}");
    }

    #[test]
    fn non_finite_values_are_rejected() {
        for text in ["0 1.0\n1 NaN\n", "0 1.0\n1 inf\n"] {
            let err = read(text).unwrap_err();
            assert!(matches!(err, DataReaderError::Parse(_)), "{text:?}: {err}");
        }
    }

    #[test]
    fn an_empty_or_all_comment_file_is_an_error() {
        for text in ["", "\n\n", "# just a comment\n"] {
            let err = read(text).unwrap_err();
            assert!(matches!(err, DataReaderError::Format(_)), "{text:?}: {err}");
        }
    }

    #[test]
    fn a_header_after_a_comment_is_still_skipped() {
        // The header used to be recognised by line index, so a leading comment
        // pushed it past the check and it was parsed as data.
        let dataset = DataReader::default()
            .with_headers(true)
            .read("# a comment\nlabel f0 f1\n0 1.0 2.0\n1 3.0 4.0\n".as_bytes())
            .unwrap();
        assert_eq!(dataset.count(), 2);
    }

    #[test]
    fn the_label_column_can_be_moved() {
        let dataset = DataReader::default()
            .with_label_column(2)
            .read("1.0 2.0 0\n3.0 4.0 1\n".as_bytes())
            .unwrap();
        assert_eq!(dataset.num_features(), 2);
        assert_eq!(dataset[0][0].value(), 1.0);
        assert_eq!(dataset[1][0].value(), 2.0);
        assert_eq!(dataset[0][0].label(), 0.0);
    }

    #[test]
    fn a_label_column_past_the_end_is_an_error() {
        let err = DataReader::default()
            .with_label_column(7)
            .read("0 1.0 2.0\n".as_bytes())
            .unwrap_err();
        assert!(matches!(err, DataReaderError::Format(_)), "{err}");
    }

    #[test]
    fn the_repository_fixtures_still_load() {
        for (name, features, labels) in [("iris.txt", 4, 2), ("hepatitis.txt", 68, 2)] {
            let dataset = DataReader::default().read_file(&fixture(name)).unwrap();
            assert_eq!(dataset.num_features(), features, "{name}");
            assert_eq!(dataset.num_labels(), labels, "{name}");
            assert!(dataset.count() > 0, "{name}");
        }
    }
}
