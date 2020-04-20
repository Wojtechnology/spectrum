use std::sync::{Arc, RwLock};
use std::thread;

use cpal::traits::{DeviceTrait, EventLoopTrait, HostTrait};
use cpal::{Sample, StreamData, UnknownTypeOutputBuffer};

mod concurrent_tee;
mod cyclical_buffer;
mod mappers;
mod transforms;

// Public exports
pub mod config;
pub mod mp3;
pub mod raw_stream;
pub mod shared_data;

use concurrent_tee::ConcurrentTee;
use config::{Config, SpectrogramConfig};
use mappers::{f32_hann_window, f32_log_normalize};
use raw_stream::RawStream;
use shared_data::SharedData;
use transforms::{
    FFTtransformer, IteratorCollector, IteratorEnumMapper, IteratorMapper, IteratorSubSequencer,
    OptionalPipelineTransformer, StutterAggregatorTranformer, Transformer, TwoChannel,
    VectorTwoChannelCombiner,
};

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
    transformer: &mut Box<dyn Transformer<Input = TwoChannel<i16>, Output = Option<Vec<f32>>>>,
    shared_data: Arc<RwLock<SharedData>>,
) -> bool {
    let sample = match (iter.next(), iter.next()) {
        (Some(l), Some(r)) => (l, r),
        _ => return false,
    };
    *cur_idx += 2;

    match transformer.transform(sample) {
        Some(output) => {
            let mut data = shared_data.write().unwrap();
            data.set_bands(output);
        }
        None => {}
    }
    return true;
}

pub fn build_single_channel_transformer(
    config: &SpectrogramConfig,
) -> Box<dyn Transformer<Input = i16, Output = Option<Box<dyn Iterator<Item = f32>>>>> {
    let stutter_transformer =
        StutterAggregatorTranformer::<i16>::new(config.buffer_size, config.stutter_size);
    let fft_transformer = FFTtransformer::<f32>::new(config.buffer_size);
    let buffer_size = config.buffer_size;
    let hann_function = move |(i, v): (usize, f32)| f32_hann_window(i, v, buffer_size);

    let casted_to_f32 = OptionalPipelineTransformer::new(
        Box::new(stutter_transformer),
        Box::new(IteratorMapper::new(|i: i16| i as f32)),
    );
    let normalized = OptionalPipelineTransformer::new(
        Box::new(casted_to_f32),
        Box::new(IteratorMapper::new(|f: f32| f / i16::max_value() as f32)),
    );
    let hanned = OptionalPipelineTransformer::new(
        Box::new(normalized),
        Box::new(IteratorEnumMapper::new(hann_function)),
    );
    Box::new(OptionalPipelineTransformer::new(
        Box::new(hanned),
        Box::new(fft_transformer),
    ))
}

pub fn build_spectrogram_transformer(
    config: &SpectrogramConfig,
    channel_size: usize,
) -> Box<dyn Transformer<Input = TwoChannel<i16>, Output = Option<Vec<f32>>>> {
    assert!(channel_size == 2, "Only two channels currently supported");

    let combined_channels = VectorTwoChannelCombiner::new(
        build_single_channel_transformer(config),
        build_single_channel_transformer(config),
    );

    let averaged_channels = OptionalPipelineTransformer::new(
        Box::new(combined_channels),
        Box::new(IteratorMapper::new(|x: TwoChannel<f32>| (x.0 + x.1) / 2.0)),
    );

    let maybe_log_normalized: Box<
        dyn Transformer<Input = TwoChannel<i16>, Output = Option<Box<dyn Iterator<Item = f32>>>>,
    > = if config.log_scaling {
        Box::new(OptionalPipelineTransformer::new(
            Box::new(averaged_channels),
            Box::new(IteratorMapper::new(f32_log_normalize)),
        ))
    } else {
        Box::new(averaged_channels)
    };

    let maybe_sub_sequenced: Box<
        dyn Transformer<Input = TwoChannel<i16>, Output = Option<Box<dyn Iterator<Item = f32>>>>,
    > = match config.band_subset {
        Some(subset) => Box::new(OptionalPipelineTransformer::new(
            maybe_log_normalized,
            Box::new(IteratorSubSequencer::new(subset.start, subset.end)),
        )),
        None => maybe_log_normalized,
    };

    Box::new(OptionalPipelineTransformer::new(
        maybe_sub_sequenced,
        Box::new(IteratorCollector::new()),
    ))
}

pub fn generate_data<D: RawStream<i16> + 'static>(mut decoder: D, config: Config) -> Vec<Vec<f32>> {
    let channels = decoder.channels();
    assert!(channels == 2, "Generating data only supports 2 channels");
    let mut transformer = build_spectrogram_transformer(&config.spectrogram, channels);
    let mut output = vec![];
    loop {
        let sample = match (decoder.next(), decoder.next()) {
            (Some(l), Some(r)) => (l, r),
            _ => break,
        };
        match transformer.transform(sample) {
            Some(bands) => output.push(bands),
            None => (),
        };
    }
    output
}

pub fn run_audio_loop<D: RawStream<i16> + 'static>(
    decoder: D,
    shared_data: Arc<RwLock<SharedData>>,
    config: Config,
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
    let channels = decoder.channels();
    assert!(channels == 2, "Audio loop only supports 2 channels");

    // Init concurrent state
    let idx_bounds_one = Arc::new(RwLock::new(IdxBounds::new()));
    let idx_bounds_two = idx_bounds_one.clone();
    let (mut decoder_a, mut decoder_b) = ConcurrentTee::new(decoder);

    thread::spawn(move || {
        let mut cur_idx: u64 = 0;
        let mut transformer = build_spectrogram_transformer(&config.spectrogram, channels);
        loop {
            let (lower_bound, upper_bound) = {
                let bounds = idx_bounds_one.read().unwrap();
                (bounds.lower_bound, bounds.upper_bound)
            };
            while cur_idx < lower_bound {
                if !push_sample(
                    &mut decoder_b,
                    &mut cur_idx,
                    &mut transformer,
                    shared_data.clone(),
                ) {
                    return;
                }
            }
            if cur_idx < upper_bound {
                if !push_sample(
                    &mut decoder_b,
                    &mut cur_idx,
                    &mut transformer,
                    shared_data.clone(),
                ) {
                    return;
                }
            }
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
