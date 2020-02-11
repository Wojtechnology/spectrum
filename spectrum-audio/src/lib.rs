use std::error::Error;
use std::fmt;
use std::io;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, SystemTime};

use cpal::traits::{DeviceTrait, EventLoopTrait, HostTrait};
use cpal::{Sample, StreamData, UnknownTypeOutputBuffer};

mod concurrent_tee;
pub mod shared_data;

use concurrent_tee::ConcurrentTee;
use shared_data::SharedData;

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

pub mod mp3 {
    use std::io::Read;

    use minimp3::{Decoder, Error as Mp3Error, Frame};

    use super::{DecoderError, RawStream};

    pub struct Mp3Decoder<R: Send> {
        decoder: Decoder<R>,
        current_frame: Frame,
        current_frame_offset: usize,
        frame_index: usize,
    }

    impl<R> Mp3Decoder<R>
    where
        R: Read + Send,
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
        R: Read + Send,
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
        R: Read + Send,
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

struct IdxBounds {
    upper_bound: u64,
    lower_bound: u64,
}

impl IdxBounds {
    fn new() -> Self {
        Self {
            upper_bound: 0,
            lower_bound: 0,
        }
    }
}

fn push_sample<I: Iterator<Item = i16>>(
    iter: &mut I,
    cur_idx: &mut u64,
    channels: usize,
    shared_data: Arc<RwLock<SharedData>>,
) -> bool {
    let mut sample = Vec::<i16>::with_capacity(channels);
    for _ in 0..channels {
        *cur_idx += 1;
        match iter.next() {
            Some(v) => sample.push(v),
            None => return false,
        }
    }
    let mut data = shared_data.write().unwrap();
    data.set_sample(sample);
    return true;
}

pub fn run_audio_loop<D: RawStream<i16> + 'static>(
    decoder: D,
    shared_data: Arc<RwLock<SharedData>>,
) {
    // Set up stream
    let host = cpal::default_host();
    let event_loop = host.event_loop();
    let device = host
        .default_output_device()
        .expect("no output device available");
    let format = find_format_with_sample_rate(&decoder, &device);
    let stream_id = event_loop.build_output_stream(&device, &format).unwrap();
    event_loop
        .play_stream(stream_id)
        .expect("failed to play_stream");

    // Audio meta and wait computation
    let sample_rate = decoder.sample_rate();
    let channels = decoder.channels();
    let wait_nanos = ((1.0 as f64) / (sample_rate as f64 * channels as f64) * (1e9 as f64)) as u64;

    // Init concurrent state
    let idx_bounds_one = Arc::new(RwLock::new(IdxBounds::new()));
    let idx_bounds_two = idx_bounds_one.clone();
    let (mut decoder_a, mut decoder_b) = ConcurrentTee::new(decoder);

    thread::spawn(move || {
        let mut cur_idx: u64 = 0;
        let mut last_time = SystemTime::now();
        loop {
            let (lower_bound, upper_bound) = {
                let bounds = idx_bounds_one.read().unwrap();
                (bounds.lower_bound, bounds.upper_bound)
            };
            while cur_idx < lower_bound {
                if !push_sample(&mut decoder_b, &mut cur_idx, channels, shared_data.clone()) {
                    break;
                }
            }
            if cur_idx < upper_bound {
                if !push_sample(&mut decoder_b, &mut cur_idx, channels, shared_data.clone()) {
                    break;
                }
            }

            // Wait to immitate progression of time
            let proc_duration = last_time.elapsed().unwrap();
            let wait_duration = Duration::from_nanos(wait_nanos);
            if wait_duration > proc_duration {
                thread::sleep(wait_duration - proc_duration);
            }
            last_time = SystemTime::now();
        }
    });

    event_loop.run(move |stream_id, stream_result| {
        let stream_data = match stream_result {
            Ok(data) => data,
            Err(err) => {
                eprintln!("an error occurred on stream {:?}: {}", stream_id, err);
                return;
            }
        };

        let mut buffer_count: u64 = 0;
        match stream_data {
            StreamData::Output {
                buffer: UnknownTypeOutputBuffer::U16(mut buffer),
            } => {
                for elem in buffer.iter_mut() {
                    let v = match decoder_a.next() {
                        Some(v) => v.to_u16(),
                        None => u16::max_value() / 2,
                    };
                    *elem = v;
                    buffer_count += 1;
                }
            }
            StreamData::Output {
                buffer: UnknownTypeOutputBuffer::I16(mut buffer),
            } => {
                for elem in buffer.iter_mut() {
                    let v = match decoder_a.next() {
                        Some(v) => v,
                        None => 0,
                    };
                    *elem = v;
                    buffer_count += 1;
                }
            }
            StreamData::Output {
                buffer: UnknownTypeOutputBuffer::F32(mut buffer),
            } => {
                for elem in buffer.iter_mut() {
                    let v = match decoder_a.next() {
                        Some(v) => v.to_f32(),
                        None => 0.0,
                    };
                    *elem = v;
                    buffer_count += 1;
                }
            }
            _ => (),
        }

        // Update bounds
        let mut bounds = idx_bounds_two.write().unwrap();
        bounds.lower_bound = bounds.upper_bound;
        bounds.upper_bound += buffer_count;
    });
}
