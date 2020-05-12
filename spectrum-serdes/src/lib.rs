use std::fs::File;

use protobuf::{CodedOutputStream, Message, RepeatedField};

mod model;

use model::data::{AudioData, AudioDataMeta, F32ChannelData, F32Channels};

#[derive(Debug)]
pub struct SaveError {
    pub path: String,
    pub msg: String,
}

impl SaveError {
    pub fn new(path: &str, msg: &str) -> Self {
        Self {
            path: String::from(path),
            msg: String::from(msg),
        }
    }
}

pub trait AudioDataWriter {
    type Elem;

    fn write(&mut self, elem: Self::Elem);

    fn save(&self, path: &str) -> Result<(), SaveError>;
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

    fn write(&mut self, elem: (f32, f32)) {
        self.buffer.push(elem);
    }

    fn save(&self, path: &str) -> Result<(), SaveError> {
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

        let mut writer = match File::create(path) {
            Ok(writer) => Ok(writer),
            Err(e) => Err(SaveError::new(path, &format!("{:?}", e))),
        }?;

        match audio_data.write_to(&mut CodedOutputStream::new(&mut writer)) {
            Ok(()) => Ok(()),
            Err(e) => Err(SaveError::new(path, &format!("{:?}", e))),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
