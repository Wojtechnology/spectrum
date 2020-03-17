use std::error::Error;
use std::fmt;
use std::io;

#[derive(Debug)]
pub enum DecoderError {
    Io(io::Error),
    Application(&'static str),
    IllFormed,
    Empty,
}

impl fmt::Display for DecoderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DecoderError::Io(io_err) => write!(f, "Error when reading file: {}", io_err),
            DecoderError::Application(s) => write!(f, "Application error: {}", s),
            DecoderError::IllFormed => write!(f, "File is illformed"),
            DecoderError::Empty => write!(f, "File is empty"),
        }
    }
}

impl Error for DecoderError {
    fn description(&self) -> &str {
        match self {
            DecoderError::Io(_) => "Error when reading file",
            DecoderError::Application(_) => "Application error",
            DecoderError::IllFormed => "File is illformed",
            DecoderError::Empty => "File is empty",
        }
    }
}

pub trait RawStream<T>: Iterator<Item = T> + Send {
    fn channels(&self) -> usize;
    fn sample_rate(&self) -> i32;
}
