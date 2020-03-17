pub struct SharedData {
    cur_channel_bands: Vec<Vec<f32>>,
}

impl SharedData {
    pub fn new(channels: usize, buffer_size: usize) -> Self {
        Self {
            cur_channel_bands: vec![vec![0.0; buffer_size]; channels],
        }
    }

    pub fn get_bands(&self) -> Vec<Vec<f32>> {
        self.cur_channel_bands.clone()
    }

    pub fn set_bands_at(&mut self, channel_idx: usize, bands: Vec<f32>) {
        self.cur_channel_bands[channel_idx] = bands;
    }
}
