use std::sync::{Arc, RwLock};
use std::thread;

use cpal::traits::{EventLoopTrait, HostTrait};
use cpal::{Sample, StreamData, UnknownTypeOutputBuffer};
use num_complex::Complex;

use crate::beat_tracking::beat_tracking_transformer;
use crate::concurrent_tee::ConcurrentTee;
use crate::config::Config;
use crate::raw_stream::{find_format_with_sample_rate, RawStream};
use crate::shared_data::SharedData;
use crate::spectrogram::{
    fft_transformer, spectrogram_viz_transformer, viz_transformer, TwoChannel,
};
use crate::transforms::Transformer;

struct AudioLoopState<I: Iterator<Item = i16>> {
    sample_iter: I,
    cur_idx: u64,
    fft_transformer:
        Box<dyn Transformer<Input = TwoChannel<i16>, Output = Option<Vec<Complex<f32>>>>>,
    viz_transformer:
        Box<dyn Transformer<Input = Box<dyn Iterator<Item = Complex<f32>>>, Output = Vec<f32>>>,
    bt_transformer:
        Box<dyn Transformer<Input = Box<dyn Iterator<Item = Complex<f32>>>, Output = bool>>,
    shared_data: Arc<RwLock<SharedData>>,
}

impl<I: Iterator<Item = i16>> AudioLoopState<I> {
    pub fn new(
        config: &Config,
        channels: usize,
        sample_iter: I,
        shared_data: Arc<RwLock<SharedData>>,
    ) -> Self {
        Self {
            sample_iter,
            cur_idx: 0,
            fft_transformer: fft_transformer(&config.spectrogram, channels),
            viz_transformer: viz_transformer(&config.spectrogram),
            bt_transformer: beat_tracking_transformer(),
            shared_data,
        }
    }

    pub fn push_next_sample(&mut self) -> bool {
        let sample = match (self.sample_iter.next(), self.sample_iter.next()) {
            (Some(l), Some(r)) => (l, r),
            _ => return false,
        };
        self.cur_idx += 2;

        match self.fft_transformer.transform(sample) {
            Some(fft_complex) => {
                let fft_complex_for_bt = fft_complex.clone();

                if self
                    .bt_transformer
                    .transform(Box::new(fft_complex_for_bt.into_iter()))
                {
                    println!("Beat: {}", self.cur_idx);
                };

                let spect_viz = self
                    .viz_transformer
                    .transform(Box::new(fft_complex.into_iter()));
                let mut data = self.shared_data.write().unwrap();
                data.set_bands(spect_viz);
            }
            None => {}
        }
        return true;
    }

    pub fn get_cur_idx(&self) -> u64 {
        self.cur_idx
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
    let (mut decoder_a, decoder_b) = ConcurrentTee::new(decoder);

    thread::spawn(move || {
        let mut state = AudioLoopState::new(&config, channels, decoder_b, shared_data);

        loop {
            let (lower_bound, upper_bound) = {
                let bounds = idx_bounds_one.read().unwrap();
                (bounds.lower_bound, bounds.upper_bound)
            };
            while state.get_cur_idx() < lower_bound {
                if !state.push_next_sample() {
                    return;
                }
            }
            if state.get_cur_idx() < upper_bound {
                if !state.push_next_sample() {
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

        // Update bounds for synchronization
        let mut bounds = idx_bounds_two.write().unwrap();
        bounds.lower_bound = bounds.upper_bound;
        bounds.upper_bound += buffer_count;
    });
}

pub fn generate_data<D: RawStream<i16> + 'static>(
    mut decoder: D,
    config: &Config,
) -> Vec<Vec<f32>> {
    let channels = decoder.channels();
    assert!(channels == 2, "Generating data only supports 2 channels");
    let mut transformer = spectrogram_viz_transformer(&config.spectrogram, channels);
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
