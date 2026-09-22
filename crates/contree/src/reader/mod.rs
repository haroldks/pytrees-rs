pub mod data_reader;

use std::error::Error;
use std::fmt;
use std::io::Error as IoError;
use std::path::Path;

/// Column delimiter of a text dataset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataFormat {
    /// Comma separated.
    Csv,
    /// Tab separated.
    Tsv,
    /// Separated by any run of whitespace.
    Space,
    /// Separated by the given character.
    Custom(char),
}

impl DataFormat {
    fn delimiter(&self) -> char {
        match self {
            DataFormat::Csv => ',',
            DataFormat::Tsv => '\t',
            DataFormat::Space => ' ',
            DataFormat::Custom(c) => *c,
        }
    }

    /// The format implied by a file extension; whitespace when unknown.
    pub fn from_extension(path: &Path) -> Self {
        match path.extension().and_then(|e| e.to_str()) {
            Some("csv") => DataFormat::Csv,
            Some("tsv") => DataFormat::Tsv,
            Some("txt") | Some("data") => DataFormat::Space,
            _ => DataFormat::Space,
        }
    }
}

/// Why a dataset file could not be read.
#[derive(Debug)]
pub enum DataReaderError {
    /// The file could not be read.
    Io(IoError),
    /// A value is not a number, or a label is not a non-negative integer.
    Parse(String),
    /// The file's shape is wrong: ragged rows, no rows, or invalid labels.
    Format(String),
}

impl fmt::Display for DataReaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataReaderError::Io(err) => write!(f, "I/O error: {}", err),
            DataReaderError::Parse(msg) => write!(f, "Parse error: {}", msg),
            DataReaderError::Format(msg) => write!(f, "Format error: {}", msg),
        }
    }
}

impl Error for DataReaderError {}

impl From<IoError> for DataReaderError {
    fn from(err: IoError) -> Self {
        DataReaderError::Io(err)
    }
}
