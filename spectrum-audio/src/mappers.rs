const F32_COMPRESSION_FACTOR: f32 = 1.0;

pub fn f32_log_compression(input: f32) -> f32 {
    (1.0 + F32_COMPRESSION_FACTOR * input).ln() / (1.0 + F32_COMPRESSION_FACTOR).ln()
}

// Not strictly a mapper, but helps with building one
pub fn f32_hann_window(index: usize, value: f32, width: usize) -> f32 {
    let hor_scale = std::f32::consts::PI / (width as f32);
    (index as f32 * hor_scale).sin().powi(2) * value
}
