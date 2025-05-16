use super::{DataFormat, DataReaderError};
use crate::bitsets::{BitCollection, Bitset, BitsetInit};
use crate::cover::Cover;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub struct DataReader {
    format: DataFormat,
    has_headers: bool,
    comment_char: Option<char>,
    label_column: Option<usize>,
}

impl Default for DataReader {
    fn default() -> Self {
        Self {
            format: DataFormat::Space,
            has_headers: false,
            comment_char: Some('#'),
            label_column: Some(0),
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

    pub fn with_label_column(mut self, label_column: Option<usize>) -> Self {
        self.label_column = label_column;
        self
    }

    pub fn auto_detect_format(mut self, path: &Path) -> Self {
        self.format = DataFormat::from_extension(path);
        self
    }

    fn read_file(&self, path: &Path) -> Result<Cover, DataReaderError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut num_cols = 0;

        let delimiter = self.format.delimiter();
        let mut row_idx = 0;

        let mut attributes: Vec<Bitset> = vec![];
        let mut target = vec![];

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

            if i <= 1 && num_cols == 0 {
                let actual_cols = if self.label_column.is_some() {
                    tokens.len() - 1
                } else {
                    tokens.len()
                };

                num_cols = num_cols.max(actual_cols);
                attributes = vec![Bitset::new(BitsetInit::Empty(64)); num_cols]
            }

            for (col_idx, &token) in tokens.iter().enumerate() {
                if Some(col_idx) == self.label_column {
                    match token.parse::<usize>() {
                        Ok(val) => target.push(val),
                        Err(_) => {
                            return Err(DataReaderError::Parse(format!(
                                "Parse error at line {}, column {}: {}",
                                i + 1,
                                col_idx + 1,
                                token
                            )));
                        }
                    }
                    continue;
                }

                let effective_col = if col_idx > self.label_column.unwrap_or(usize::MAX) {
                    col_idx - 1
                } else {
                    col_idx
                };

                let capacity = attributes[effective_col].capacity();

                if row_idx >= capacity {
                    attributes[effective_col].resize(capacity * 2);
                }

                match token.parse::<usize>() {
                    Ok(0) => {}

                    Ok(1) => {
                        attributes[effective_col].set(row_idx);
                    }

                    Ok(_) => {
                        return Err(DataReaderError::Format(format!(
                            "Non-binary value at line {}, column {}",
                            i + 1,
                            col_idx + 1
                        )))
                    }

                    Err(_) => {
                        return Err(DataReaderError::Parse(format!(
                            "Parse error at line {}, column {}: {}",
                            i + 1,
                            col_idx + 1,
                            token
                        )));
                    }
                }
            }

            row_idx += 1;
        }

        for bitset in attributes.iter_mut() {
            bitset.resize(row_idx);
        }

        let unique_targets = target.iter().map(|t| *t).collect::<HashSet<usize>>().len();

        let mut targets = vec![Bitset::new(BitsetInit::Empty(row_idx)); unique_targets];

        for (tid, &t) in target.iter().enumerate() {
            targets[t].set(tid);
        }
        Ok(Cover::new(attributes, targets, row_idx))
    }
}

#[cfg(test)]
mod data_reader_test {
    use std::path::Path;
    use crate::cover::Cover;
    use crate::reader::data_reader::DataReader;
    use crate::reader::DataReaderError;

    #[test]
    fn load_small() {
        
        let reader = DataReader::default();
        let path = Path::new("test_data/small_.txt");
        let cover_result = reader.read_file(path);
        let cover  = match cover_result {
            Ok(cover) => {cover}
            Err(err) => {
                println!("Data error {}", err );
                panic!("oops")
            }
        };
        
        assert_eq!(cover.num_labels, 2);
        assert_eq!(cover.num_attributes, 4);
        assert_eq!(cover.count(), 10);
        
        assert_eq!(cover.to_vec().iter().eq((0..10).collect::<Vec<usize>>().iter()), true)
        
    }
    
    
}
