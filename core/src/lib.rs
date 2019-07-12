extern crate minimp3;

use std::io;
use std::io::Read;

#[derive(Debug)]
pub enum DecoderError {
    Io(io::Error),
    Application(&'static str),
    IllFormed,
    Empty,
}

pub mod mp3 {
    use minimp3::{Decoder, Error as Mp3Error, Frame};

    use super::{DecoderError, Read};

    pub struct Mp3Decoder<R> {
        decoder: Decoder<R>,
        current_frame: Frame,
        current_frame_offset: usize,
    }

    impl<R> Mp3Decoder<R>
    where
        R: Read,
    {
        pub fn new(reader: R) -> Result<Mp3Decoder<R>, DecoderError> {
            let mut decoder = Decoder::new(reader);
            let current_frame = loop {
                match decoder.next_frame() {
                    Ok(frame) => break Ok(frame),
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
            })
        }
    }

    impl<R> Iterator for Mp3Decoder<R>
    where
        R: Read,
    {
        type Item = i16;

        fn next(&mut self) -> Option<i16> {
            if self.current_frame_offset == self.current_frame.data.len() {
                self.current_frame_offset = 0;
                match self.decoder.next_frame() {
                    Ok(frame) => self.current_frame = frame,
                    Err(_) => return None,
                }
            }

            let v = self.current_frame.data[self.current_frame_offset];
            self.current_frame_offset += 1;
            Some(v)
        }
    }
}
