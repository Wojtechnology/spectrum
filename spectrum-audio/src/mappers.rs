const F32_LOG_C: f32 = 1e-7;

pub fn f32_log_normalize(input: f32) -> f32 {
    (input + F32_LOG_C).ln()
}

// Not strictly a mapper, but helps with building one
pub fn f32_hann_window(index: usize, value: f32, width: usize) -> f32 {
    let hor_scale = std::f32::consts::PI / (width as f32);
    (index as f32 * hor_scale).sin().powi(2) * value
}
