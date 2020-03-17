use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, SystemTime};

use cpal::traits::{DeviceTrait, EventLoopTrait, HostTrait};
use cpal::{Sample, StreamData, UnknownTypeOutputBuffer};

use crate::transforms::{
    FFTtransformer, I16ToF32Transformer, OptionalPipelineTransformer, StutterAggregatorTranformer,
    Transformer, VectorTransformer,
};

mod concurrent_tee;
mod cyclical_buffer;
mod transforms;

// Public exports
pub mod mp3;
pub mod raw_stream;
pub mod shared_data;

use concurrent_tee::ConcurrentTee;
use raw_stream::RawStream;
use shared_data::SharedData;

// TODO: Make these configuration
pub const BUFFER_SIZE: usize = 1024;
const STUTTER_SIZE: usize = BUFFER_SIZE / 2;

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
    channel_transformers: &mut Vec<Box<dyn Transformer<Input = i16, Output = Option<Vec<f32>>>>>,
    shared_data: Arc<RwLock<SharedData>>,
) -> bool {
    for (channel_idx, channel_transformer) in channel_transformers.iter_mut().enumerate() {
        *cur_idx += 1;
        match iter.next() {
            Some(input) => match channel_transformer.transform(input) {
                Some(output) => {
                    let mut data = shared_data.write().unwrap();
                    data.set_bands_at(channel_idx, output);
                }
                None => (),
            },
            None => return false,
        }
    }
    return true;
}

fn build_transformer(
    buffer_size: usize,
    stutter_size: usize,
) -> Box<dyn Transformer<Input = i16, Output = Option<Vec<f32>>>> {
    let stutter_transformer = StutterAggregatorTranformer::<i16>::new(buffer_size, stutter_size);
    let i16_to_f32_transformer = I16ToF32Transformer::new();
    let itf_vec_tranformer = VectorTransformer::new(Box::new(i16_to_f32_transformer));
    let fft_transformer = FFTtransformer::<f32>::new(buffer_size);
    let opt_pl_transformer_one = OptionalPipelineTransformer::new(
        Box::new(stutter_transformer),
        Box::new(itf_vec_tranformer),
    );
    let opt_pl_transformer_two = OptionalPipelineTransformer::new(
        Box::new(opt_pl_transformer_one),
        Box::new(fft_transformer),
    );
    Box::new(opt_pl_transformer_two)
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

        let mut channel_transformers = Vec::with_capacity(channels);
        for _ in 0..channels {
            channel_transformers.push(build_transformer(BUFFER_SIZE, STUTTER_SIZE));
        }

        loop {
            let (lower_bound, upper_bound) = {
                let bounds = idx_bounds_one.read().unwrap();
                (bounds.lower_bound, bounds.upper_bound)
            };
            while cur_idx < lower_bound {
                if !push_sample(
                    &mut decoder_b,
                    &mut cur_idx,
                    &mut channel_transformers,
                    shared_data.clone(),
                ) {
                    return;
                }
            }
            if cur_idx < upper_bound {
                if !push_sample(
                    &mut decoder_b,
                    &mut cur_idx,
                    &mut channel_transformers,
                    shared_data.clone(),
                ) {
                    return;
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
