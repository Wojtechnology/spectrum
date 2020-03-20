pub struct SharedData {
    cur_channel_bands: Vec<f32>,
}

impl SharedData {
    pub fn new(num_vals: usize) -> Self {
        Self {
            cur_channel_bands: vec![0.0; num_vals],
        }
    }

    pub fn get_bands(&self) -> Vec<f32> {
        self.cur_channel_bands.clone()
    }

    pub fn set_bands(&mut self, bands: Vec<f32>) {
        self.cur_channel_bands = bands;
    }
}
