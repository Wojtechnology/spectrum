use std::error::Error;
use std::fmt;
use std::io;

use cpal::traits::DeviceTrait;

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

// Helpers

pub fn find_format_with_sample_rate<D: RawStream<i16>>(
    decoder: &D,
    device: &cpal::Device,
) -> cpal::Format {
    let mut supported_formats_range = device
        .supported_output_formats()
        .expect("error while querying_formats");
    let decoder_sample_rate = decoder.sample_rate() as u32;
    loop {
        let supported_format = supported_formats_range
            .next()
            .expect("Couldn't find a suitable output format");
        if decoder_sample_rate >= supported_format.min_sample_rate.0
            && decoder_sample_rate <= supported_format.max_sample_rate.0
        {
            break cpal::Format {
                channels: supported_format.channels,
                sample_rate: cpal::SampleRate(decoder_sample_rate),
                data_type: supported_format.data_type,
            };
        }
    }
}
