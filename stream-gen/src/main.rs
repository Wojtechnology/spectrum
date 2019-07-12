extern crate core;

use std::fs::File;

use core::mp3::Mp3Decoder;

fn main() {
    // TODO: Error handling
    let reader = File::open("/Users/wojtekswiderski/Music/iTunes/iTunes Media/Music/Felix Cartel/Unknown Album/Young Love.mp3").unwrap();
    let decoder = Mp3Decoder::new(reader).unwrap();
    for _ in decoder {}
}
