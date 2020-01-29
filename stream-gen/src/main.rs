extern crate spectrum_audio;

use std::env;
use std::fs::File;
use std::process;

use spectrum_audio::mp3::Mp3Decoder;
use spectrum_audio::RawStream;

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

    println!(
        "Track {}\n\tchannels: {}\n\tsample_rate: {}",
        &filename,
        decoder.channels(),
        decoder.sample_rate()
    );

    for _ in decoder {}
}
