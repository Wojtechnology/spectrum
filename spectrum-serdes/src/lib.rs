use std::io::{Read, Write};
use std::marker::PhantomData;

use protobuf::{CodedInputStream, Message, RepeatedField};

mod model;

use model::data::{AudioData, AudioDataMeta, F32ChannelData, F32Channels};

#[derive(Debug)]
pub struct SaveError {
    pub msg: String,
}

impl SaveError {
    pub fn new(msg: &str) -> Self {
        Self {
            msg: String::from(msg),
        }
    }
}

#[derive(Debug)]
pub struct ReadError {
    pub msg: String,
}

impl ReadError {
    pub fn new(msg: &str) -> Self {
        Self {
            msg: String::from(msg),
        }
    }
}

pub trait AudioDataWriter {
    type Elem;

    fn push(&mut self, elem: Self::Elem);

    fn write(&self, writer: &mut dyn Write) -> Result<(), SaveError>;
}

pub struct TwoChannelWriter<T> {
    sample_rate: f64,
    offset: i64,
    buffer: Vec<(T, T)>,
}

impl<T> TwoChannelWriter<T> {
    pub fn new(sample_rate: f64, offset: i64) -> Self {
        Self {
            sample_rate,
            offset,
            buffer: Vec::new(),
        }
    }
}

impl AudioDataWriter for TwoChannelWriter<f32> {
    type Elem = (f32, f32);

    fn push(&mut self, elem: (f32, f32)) {
        self.buffer.push(elem);
    }

    fn write(&self, writer: &mut dyn Write) -> Result<(), SaveError> {
        let mut meta = AudioDataMeta::new();
        meta.set_size(self.buffer.len() as u64);
        meta.set_channels(2);
        meta.set_sample_rate(self.sample_rate);
        meta.set_offset(self.offset);

        let mut l_vals = Vec::new();
        let mut r_vals = Vec::new();
        for &(l_val, r_val) in self.buffer.iter() {
            l_vals.push(l_val);
            r_vals.push(r_val);
        }
        let mut l_data = F32ChannelData::new();
        let mut r_data = F32ChannelData::new();
        l_data.set_values(l_vals);
        r_data.set_values(r_vals);

        let mut channels = F32Channels::new();
        channels.set_channels(RepeatedField::from_vec(vec![l_data, r_data]));

        let mut audio_data = AudioData::new();
        audio_data.set_meta(meta);
        audio_data.set_f32_channels(channels);

        match audio_data.write_to_writer(writer) {
            Ok(()) => Ok(()),
            Err(e) => Err(SaveError::new(&format!("{:?}", e))),
        }
    }
}

impl AudioData {
    pub fn from_reader(reader: &mut dyn Read) -> Result<Self, ReadError> {
        let mut data = AudioData::new();
        match data.merge_from(&mut CodedInputStream::new(reader)) {
            Ok(()) => Ok(data),
            Err(e) => Err(ReadError::new(&format!("{:?}", e))),
        }
    }
}

// pub struct TwoChannelReader<'a, T> {
//     data: &'a AudioData,
//     cursor: usize,
//     phantom: PhantomData<T>,
// }
//
// impl<'a, T> Iterator for TwoChannelReader<'a, T> {
//     type Item = (T, T);
//
//     fn next(&mut self) -> Option<(T, T)> {
//         let channels
//         if cursor <
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_reads_and_writes_two_channel() {
        let mut buf = Vec::<u8>::new();
        let mut writer = TwoChannelWriter::new(44100.0, -100);
        writer.push((123.0, 456.0));
        writer.push((234.0, 345.0));
        let write_res = writer.write(&mut buf);
        write_res.expect("writer.write should be ok");

        let audio_data = AudioData::from_reader(&mut &buf[..]).expect("should be able to read");
        let meta = audio_data.get_meta();
        assert_eq!(meta.get_size(), 2);
        assert_eq!(meta.get_channels(), 2);
        assert_eq!(meta.get_sample_rate(), 44100.0);
        assert_eq!(meta.get_offset(), -100);
        assert_eq!(audio_data.get_f32_channels().get_channels().len(), 2);
        assert_eq!(
            audio_data.get_f32_channels().get_channels()[0].get_values(),
            &[123.0, 234.0],
        );
        assert_eq!(
            audio_data.get_f32_channels().get_channels()[1].get_values(),
            &[456.0, 345.0],
        );
        assert!(!audio_data.has_f64_channels(), "must not have f64 channels");
    }
}
