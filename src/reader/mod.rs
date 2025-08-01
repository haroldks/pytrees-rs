mod data_info;
pub mod data_reader;

use std::error::Error;
use std::fmt;

use std::io::Error as IoError;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataFormat {
    Csv,
    Tsv,
    Space,
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

    pub fn from_extension(path: &Path) -> Self {
        match path.extension().and_then(|e| e.to_str()) {
            Some("csv") => DataFormat::Csv,
            Some("tsv") => DataFormat::Tsv,
            Some("txt") | Some("data") => DataFormat::Space,
            _ => DataFormat::Space,
        }
    }
}

#[derive(Debug)]
pub enum DataReaderError {
    Io(IoError),
    Parse(String),
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
