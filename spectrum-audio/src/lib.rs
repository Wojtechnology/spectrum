use std::error::Error;
use std::fmt;
use std::io;

use cpal::traits::{DeviceTrait, EventLoopTrait, HostTrait};
use cpal::{Sample, StreamData, UnknownTypeOutputBuffer};

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

pub trait RawStream<T>: Iterator<Item = T> {
    fn channels(&self) -> usize;
    fn sample_rate(&self) -> i32;
}

pub mod mp3 {
    use std::io::Read;

    use minimp3::{Decoder, Error as Mp3Error, Frame};

    use super::{DecoderError, RawStream};

    pub struct Mp3Decoder<R> {
        decoder: Decoder<R>,
        current_frame: Frame,
        current_frame_offset: usize,
        frame_index: usize,
    }

    impl<R> Mp3Decoder<R>
    where
        R: Read,
    {
        pub fn new(reader: R) -> Result<Mp3Decoder<R>, DecoderError> {
            let mut decoder = Decoder::new(reader);
            let current_frame = loop {
                match decoder.next_frame() {
                    Ok(frame) => {
                        break Ok(frame);
                    }
                    Err(Mp3Error::Io(err)) => break Err(DecoderError::Io(err)),
                    Err(Mp3Error::InsufficientData) => {
                        break Err(DecoderError::Application("Insufficient data"))
                    }
                    Err(Mp3Error::Eof) => break Err(DecoderError::Empty),
                    Err(Mp3Error::SkippedData) => continue,
                }
            }?;

            Ok(Mp3Decoder {
                decoder,
                current_frame,
                current_frame_offset: 0,
                frame_index: 0,
            })
        }
    }

    impl<R> Iterator for Mp3Decoder<R>
    where
        R: Read,
    {
        type Item = i16;

        fn next(&mut self) -> Option<Self::Item> {
            if self.current_frame_offset == self.current_frame.data.len() {
                self.current_frame_offset = 0;
                match self.decoder.next_frame() {
                    Ok(frame) => {
                        // NOTE: We assume that the sample rate and channels of the song stays the
                        // same throughout. Should add a debug test here to check if this is
                        // actually the case
                        self.current_frame = frame;
                        self.frame_index += 1;
                    }
                    Err(_) => return None,
                }
            }

            let v = self.current_frame.data[self.current_frame_offset];
            self.current_frame_offset += 1;
            Some(v)
        }
    }

    impl<R> RawStream<i16> for Mp3Decoder<R>
    where
        R: Read,
    {
        fn channels(&self) -> usize {
            self.current_frame.channels
        }

        fn sample_rate(&self) -> i32 {
            self.current_frame.sample_rate
        }
    }
}

fn find_format_with_sample_rate<D: RawStream<i16>>(
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

pub fn run_audio_loop<D: RawStream<i16>>(decoder: D) {
    let host = cpal::default_host();
    let event_loop = host.event_loop();
    let device = host
        .default_output_device()
        .expect("no output device available");
    let format = find_format_with_sample_rate(&decoder, &device);
    println!("Using output format: {:?}", format);
    let stream_id = event_loop.build_output_stream(&device, &format).unwrap();
    event_loop
        .play_stream(stream_id)
        .expect("failed to play_stream");
    let vals: Vec<i16> = decoder.collect();
    let mut v_iter = vals.iter();
    event_loop.run(move |stream_id, stream_result| {
        let stream_data = match stream_result {
            Ok(data) => data,
            Err(err) => {
                eprintln!("an error occurred on stream {:?}: {}", stream_id, err);
                return;
            }
        };

        match stream_data {
            StreamData::Output {
                buffer: UnknownTypeOutputBuffer::U16(mut buffer),
            } => {
                for elem in buffer.iter_mut() {
                    let v = match v_iter.next() {
                        Some(&v) => v.to_u16(),
                        None => u16::max_value() / 2,
                    };
                    *elem = v;
                }
            }
            StreamData::Output {
                buffer: UnknownTypeOutputBuffer::I16(mut buffer),
            } => {
                for elem in buffer.iter_mut() {
                    let v = match v_iter.next() {
                        Some(&v) => v,
                        None => 0,
                    };
                    *elem = v;
                }
            }
            StreamData::Output {
                buffer: UnknownTypeOutputBuffer::F32(mut buffer),
            } => {
                for elem in buffer.iter_mut() {
                    let v = match v_iter.next() {
                        Some(&v) => v.to_f32(),
                        None => 0.0,
                    };
                    *elem = v;
                }
            }
            _ => (),
        }
    });
}
