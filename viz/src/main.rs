extern crate cpal;
extern crate spectrum_audio;

use std::env;
use std::fs::File;
use std::process;

use spectrum_audio::mp3::Mp3Decoder;
use spectrum_audio::RawStream;

use cpal::traits::{DeviceTrait, EventLoopTrait, HostTrait};
use cpal::{Sample, StreamData, UnknownTypeOutputBuffer};

fn run_audio_loop<D: RawStream<i16>>(decoder: D) {
    let host = cpal::default_host();
    let event_loop = host.event_loop();
    // TODO: Error processing
    let device = host
        .default_output_device()
        .expect("no output device available");
    let mut supported_formats_range = device
        .supported_output_formats()
        .expect("error while querying_formats");
    supported_formats_range.next();
    let format = supported_formats_range
        .next()
        .expect("no supported format")
        .with_max_sample_rate();
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

fn main() {
    let mut args = env::args();
    args.next();

    let filename = match args.next() {
        Some(arg) => arg,
        None => {
            eprintln!("Missing filename argument");
            process::exit(1);
        }
    };

    let reader = File::open(&filename).unwrap_or_else(|err| {
        eprintln!("Error when reading file: {}", err);
        process::exit(1);
    });

    let decoder = Mp3Decoder::new(reader).unwrap_or_else(|err| {
        eprintln!("{}", err);
        process::exit(1);
    });

    run_audio_loop(decoder);
}
