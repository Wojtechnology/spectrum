use crate::cyclical_buffer::CyclicalBuffer;
use crate::transforms::{FFTtransformer, Transformer};

use num_complex::Complex;

pub struct SharedData {
    channel_bufs: Vec<CyclicalBuffer<i16>>,
    starting_cursor: usize,
    stutter_cursor: usize,
    transformer: FFTtransformer<f32>,
    cur_bands: Vec<Vec<f32>>,
}

// TODO: Make these configuration
pub const BUFFER_SIZE: usize = 1024;
const STUTTER_SIZE: usize = BUFFER_SIZE / 2;

impl SharedData {
    pub fn new(channels: usize) -> Self {
        let mut channel_bufs = Vec::<CyclicalBuffer<i16>>::with_capacity(channels);
        for _ in 0..channels {
            channel_bufs.push(CyclicalBuffer::new(BUFFER_SIZE));
        }
        Self {
            channel_bufs,
            starting_cursor: 0,
            stutter_cursor: 0,
            transformer: FFTtransformer::new(BUFFER_SIZE),
            cur_bands: vec![vec![0.0; BUFFER_SIZE]; channels],
        }
    }

    fn update_cur_bands(&mut self) {
        self.cur_bands = Vec::with_capacity(self.channel_bufs.len());
        for buf in self.channel_bufs.iter() {
            let input: Vec<_> = buf
                .get_values()
                .iter()
                .map(|&v| Complex::new(v as f32, 0.0))
                .collect();
            let output: Vec<_> = self
                .transformer
                .transform(input)
                .iter()
                .map(|c| (c.re * c.re + c.im * c.im).sqrt())
                .collect();
            self.cur_bands.push(output);
        }
    }

    pub fn get_bands(&self) -> Vec<Vec<f32>> {
        self.cur_bands.clone()
    }

    pub fn set_sample(&mut self, sample: Vec<i16>) {
        assert!(
            sample.len() == self.channel_bufs.len(),
            "Size of sample must be the same as number of channels"
        );
        for (idx, v) in sample.iter().enumerate() {
            self.channel_bufs[idx].push(*v);
        }
        if self.starting_cursor < BUFFER_SIZE {
            self.starting_cursor += 1;
        } else {
            if self.stutter_cursor == 0 {
                self.update_cur_bands();
            }
            self.stutter_cursor = (self.stutter_cursor + 1) % STUTTER_SIZE;
        }
    }
}
