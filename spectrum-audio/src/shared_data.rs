pub struct SharedData {
    sample: Vec<i16>,
}

const ALPHA: f32 = 0.1;

impl SharedData {
    pub fn new(channels: usize) -> Self {
        let mut sample = Vec::<i16>::with_capacity(channels);
        for _ in 0..channels {
            sample.push(0);
        }
        Self { sample }
    }

    pub fn get_sample(&self) -> Vec<i16> {
        self.sample.clone()
    }

    pub fn set_sample(&mut self, sample: Vec<i16>) {
        // Size of sample must be the same as self.sample (i.e. same number of channels).
        for (idx, v) in sample.iter().enumerate() {
            self.sample[idx] =
                (((1.0 - ALPHA) * (self.sample[idx] as f32)) + (ALPHA * (*v as f32))) as i16;
        }
    }
}
