use num_complex::Complex;

const F32_COMPRESSION_FACTOR: f32 = 1.0;

pub fn log_compression(input: f32) -> f32 {
    (1.0 + F32_COMPRESSION_FACTOR * input).ln() / (1.0 + F32_COMPRESSION_FACTOR).ln()
}

pub fn hann_window(index: usize, value: f32, width: usize) -> f32 {
    let hor_scale = std::f32::consts::PI / (width as f32);
    (index as f32 * hor_scale).sin().powi(2) * value
}

pub fn complex_l1_norm(c: Complex<f32>) -> f32 {
    c.re.abs() + c.im.abs()
}

pub fn complex_l2_norm(c: Complex<f32>) -> f32 {
    (c.re * c.re + c.im * c.im).sqrt()
}

pub fn spectral_flux(it: Box<dyn Iterator<Item = f32>>) -> f32 {
    return it.fold(0.0, |acc, x| acc + x.max(0.0));
}
