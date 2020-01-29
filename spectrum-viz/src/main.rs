use std::env;
use std::fs::File;
use std::process;
use std::thread;

use spectrum_audio::mp3::Mp3Decoder;
use spectrum_audio::run_audio_loop;
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

    thread::spawn(move || {
        run_audio_loop(decoder);
    });
    event_loop::run();
}
