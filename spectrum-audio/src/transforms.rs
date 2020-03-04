use std::sync::Arc;

use num_complex::Complex;
use num_traits::Zero;
use rustfft::{FFTnum, FFTplanner, FFT};

pub trait Transformer {
    type Input;
    type Output;

    fn transform(&self, input: Self::Input) -> Self::Output;
}

// BEGIN: FFTtransformer

pub struct FFTtransformer<T: FFTval> {
    fft: Arc<dyn FFT<T>>,
}

impl<T: FFTval> FFTtransformer<T> {
    pub fn new(size: usize) -> Self {
        Self {
            fft: FFTplanner::new(false).plan_fft(size),
        }
    }
}

pub trait FFTval: FFTnum + Clone {
    fn norm_divisor(size: usize) -> Self;
}

impl FFTval for f32 {
    fn norm_divisor(size: usize) -> f32 {
        (size as f32).sqrt()
    }
}

impl FFTval for f64 {
    fn norm_divisor(size: usize) -> f64 {
        (size as f64).sqrt()
    }
}

impl<T: FFTval> Transformer for FFTtransformer<T> {
    type Input = Vec<Complex<T>>;
    type Output = Vec<Complex<T>>;

    fn transform(&self, input: Vec<Complex<T>>) -> Vec<Complex<T>> {
        let mut m_input = input.clone();
        let mut m_output = vec![Complex::zero(); self.fft.len()];
        self.fft.process(&mut m_input, &mut m_output);

        // normalize
        let len_sqrt = T::norm_divisor(self.fft.len());
        m_output.into_iter().map(|elem| elem / len_sqrt).collect()
    }
}

// END: FFTtransformer
