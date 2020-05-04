#[allow(dead_code)]
pub struct RunningStat {
    n: usize,
    m: f64,
    s: f64,
}

#[allow(dead_code)]
impl RunningStat {
    pub fn new() -> Self {
        Self {
            n: 0,
            m: 0.0,
            s: 0.0,
        }
    }

    pub fn push_f32(&mut self, x: f32) {
        self.push_f64(x as f64)
    }

    pub fn push_f64(&mut self, x: f64) {
        self.n += 1;

        if self.n == 1 {
            self.m = x;
            self.s = 0.0;
        } else {
            let old_m = self.m;
            let old_s = self.s;
            self.m = old_m + (x - old_m) / (self.n as f64);
            self.s = old_s + (x - old_m) * (x - self.m);
        }
    }

    pub fn cur_mean(&self) -> f64 {
        if self.n > 0 {
            self.m
        } else {
            0.0
        }
    }

    pub fn cur_stdev(&self) -> f64 {
        if self.n > 1 {
            (self.s / (self.n - 1) as f64).sqrt()
        } else {
            0.0
        }
    }
}
