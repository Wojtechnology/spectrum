use std::env;
use std::fs::File;
use std::process;
use std::sync::{Arc, RwLock};
use std::thread;

use spectrum_audio::mp3::Mp3Decoder;
use spectrum_audio::raw_stream::RawStream;
use spectrum_audio::run_audio_loop;
use spectrum_audio::shared_data::SharedData;
use spectrum_audio::BUFFER_SIZE;
use spectrum_viz::event_loop;

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

    let shared_data = Arc::new(RwLock::new(SharedData::new(128)));
    let shared_data_clone = shared_data.clone();

    thread::spawn(move || {
        run_audio_loop(decoder, shared_data_clone);
    });
    event_loop::run(shared_data);
}
