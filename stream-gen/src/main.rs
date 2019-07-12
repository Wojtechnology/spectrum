extern crate core;

use std::env;
use std::fs::File;
use std::process;

use core::mp3::Mp3Decoder;
use core::DecoderError;

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

    let reader = File::open(filename).unwrap_or_else(|err| {
        eprintln!("Error when reading file: {}", err);
        process::exit(1);
    });
    let decoder = Mp3Decoder::new(reader).unwrap_or_else(|err| {
        let err_msg = match err {
            DecoderError::Io(io_err) => format!("Error when reading file: {}", io_err),
            DecoderError::Application(s) => format!("Application error: {}", s),
            DecoderError::IllFormed => format!("File is illformed"),
            DecoderError::Empty => format!("File is empty"),
        };
        eprintln!("{}", err_msg);
        process::exit(1);
    });
    for _ in decoder {}
}
