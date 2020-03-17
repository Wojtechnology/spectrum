use std::sync::Arc;

use num_complex::Complex;
use num_traits::Zero;
use rustfft::{FFTnum, FFTplanner, FFT};

use crate::cyclical_buffer::CyclicalBuffer;

pub trait Transformer {
    type Input;
    type Output;

    fn transform(&mut self, input: Self::Input) -> Self::Output;
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
    fn cast_and_sqrt(size: usize) -> Self;
    fn sqrt(f: Self) -> Self;
    fn real_to_complex(f: Self) -> Complex<Self>;
}

impl FFTval for f32 {
    fn cast_and_sqrt(size: usize) -> f32 {
        (size as f32).sqrt()
    }

    fn sqrt(f: f32) -> f32 {
        f.sqrt()
    }

    fn real_to_complex(f: f32) -> Complex<f32> {
        Complex::new(f, 0.0)
    }
}

impl FFTval for f64 {
    fn cast_and_sqrt(size: usize) -> f64 {
        (size as f64).sqrt()
    }

    fn sqrt(f: f64) -> f64 {
        f.sqrt()
    }

    fn real_to_complex(f: f64) -> Complex<f64> {
        Complex::new(f, 0.0)
    }
}

impl<T: FFTval> Transformer for FFTtransformer<T> {
    type Input = Vec<T>;
    type Output = Vec<T>;

    fn transform(&mut self, input: Vec<T>) -> Vec<T> {
        let mut m_input: Vec<_> = input.iter().map(|&v| T::real_to_complex(v)).collect();
        let mut m_output = vec![Complex::zero(); self.fft.len()];
        self.fft.process(&mut m_input, &mut m_output);

        // normalize
        let len_sqrt = T::cast_and_sqrt(self.fft.len());
        m_output
            .into_iter()
            .map(|elem| {
                let c = elem / len_sqrt;
                T::sqrt(c.re * c.re + c.im * c.im)
            })
            .collect()
    }
}

// END: FFTtransformer

// BEGIN: StutterAggregatorTranformer

pub struct StutterAggregatorTranformer<T: Copy> {
    buf: CyclicalBuffer<T>,
    buffer_size: usize,
    stutter_size: usize,
    starting_cursor: usize,
    stutter_cursor: usize,
}

impl<T: Copy> StutterAggregatorTranformer<T> {
    pub fn new(buffer_size: usize, stutter_size: usize) -> Self {
        Self {
            buf: CyclicalBuffer::new(buffer_size),
            buffer_size,
            stutter_size,
            starting_cursor: 0,
            stutter_cursor: 0,
        }
    }
}

impl<T: Copy> Transformer for StutterAggregatorTranformer<T> {
    type Input = T;
    type Output = Option<Vec<T>>;

    fn transform(&mut self, input: T) -> Option<Vec<T>> {
        self.buf.push(input);
        if self.starting_cursor < self.buffer_size {
            self.starting_cursor += 1;
            None
        } else {
            let cur_stutter_cursor = self.stutter_cursor;
            self.stutter_cursor = (self.stutter_cursor + 1) % self.stutter_size;
            if cur_stutter_cursor == 0 {
                Some(self.buf.get_values())
            } else {
                None
            }
        }
    }
}

// END: StutterAggregatorTranformer

// BEGIN: I16ToF32Transformer

pub struct I16ToF32Transformer {}

impl I16ToF32Transformer {
    pub fn new() -> Self {
        Self {}
    }
}

impl Transformer for I16ToF32Transformer {
    type Input = i16;
    type Output = f32;

    fn transform(&mut self, input: i16) -> f32 {
        input as f32
    }
}

// END: I16ToF32Transformer

// BEGIN: VectorTransformer

pub struct VectorTransformer<I: Copy, O: Copy> {
    transformer: Box<dyn Transformer<Input = I, Output = O>>,
}

impl<I: Copy, O: Copy> VectorTransformer<I, O> {
    pub fn new(transformer: Box<dyn Transformer<Input = I, Output = O>>) -> Self {
        Self { transformer }
    }
}

impl<I: Copy, O: Copy> Transformer for VectorTransformer<I, O> {
    type Input = Vec<I>;
    type Output = Vec<O>;

    fn transform(&mut self, input: Vec<I>) -> Vec<O> {
        input
            .iter()
            .map(|v| self.transformer.transform(*v))
            .collect()
    }
}

// END: VectorTransformer

// BEGIN: OptionalPipelineTransformer

pub struct OptionalPipelineTransformer<I, M, O> {
    transformer_one: Box<dyn Transformer<Input = I, Output = Option<M>>>,
    transformer_two: Box<dyn Transformer<Input = M, Output = O>>,
}

impl<I, M, O> OptionalPipelineTransformer<I, M, O> {
    pub fn new(
        transformer_one: Box<dyn Transformer<Input = I, Output = Option<M>>>,
        transformer_two: Box<dyn Transformer<Input = M, Output = O>>,
    ) -> Self {
        Self {
            transformer_one,
            transformer_two,
        }
    }
}

impl<I, M, O> Transformer for OptionalPipelineTransformer<I, M, O> {
    type Input = I;
    type Output = Option<O>;

    fn transform(&mut self, input: I) -> Option<O> {
        match self.transformer_one.transform(input) {
            Some(mid_output) => Some(self.transformer_two.transform(mid_output)),
            None => None,
        }
    }
}

// END: OptionalPipelineTransformer
