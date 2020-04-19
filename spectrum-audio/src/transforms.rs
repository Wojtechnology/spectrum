use std::marker::PhantomData;
use std::sync::Arc;
use std::time::SystemTime;

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
        let start = SystemTime::now();
        let mut m_input: Vec<_> = input.iter().map(|&v| T::real_to_complex(v)).collect();
        let mut m_output = vec![Complex::zero(); self.fft.len()];
        self.fft.process(&mut m_input, &mut m_output);

        // normalize
        let len_sqrt = T::cast_and_sqrt(self.fft.len());
        let out = m_output
            .into_iter()
            .map(|elem| {
                let c = elem / len_sqrt;
                T::sqrt(c.re * c.re + c.im * c.im)
            })
            .collect();
        println!("FFTTime({})", start.elapsed().unwrap().as_micros());
        out
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

// BEGIN: F32ExponentialSmoothingTransformer

pub enum SmoothingKernel {
    Average,
    Difference,
}

pub struct F32ExponentialSmoothingTransformer {
    damping_factor: f32,
    lead_in: usize,
    kernel: SmoothingKernel,
}

impl F32ExponentialSmoothingTransformer {
    pub fn new(damping_factor: f32, lead_in: usize, kernel: SmoothingKernel) -> Self {
        assert!(
            damping_factor >= 0.0,
            "damping_factor must be greater than or equal to 0"
        );
        assert!(
            damping_factor <= 1.0,
            "damping_factor must be less than or equal to 1"
        );
        Self {
            damping_factor,
            lead_in,
            kernel,
        }
    }
}

impl Transformer for F32ExponentialSmoothingTransformer {
    type Input = Vec<f32>;
    type Output = Vec<f32>;

    fn transform(&mut self, input: Vec<f32>) -> Vec<f32> {
        let input_len = input.len();
        assert!(
            input_len > 2 * self.lead_in,
            "input length must be more than twice lead_in"
        );
        let output_len = input_len - 2 * self.lead_in;

        let mut f_avg = vec![0.0; output_len];
        let mut b_avg = vec![0.0; output_len];
        let mut cur_f = input[0];
        let mut cur_b = input[input_len - 1];
        for i in 1..(input_len - self.lead_in) {
            cur_f = input[i] * self.damping_factor + (1.0 - self.damping_factor) * cur_f;
            cur_b = input[input_len - 1 - i] * self.damping_factor
                + (1.0 - self.damping_factor) * cur_b;
            if i >= self.lead_in {
                f_avg[i - self.lead_in] = cur_f;
                b_avg[output_len - 1 - (i - self.lead_in)] = cur_b;
            }
        }
        let mut output = Vec::with_capacity(output_len);
        for i in 0..output_len {
            output.push(match self.kernel {
                SmoothingKernel::Average => (f_avg[i] + b_avg[i]) / 2.0,
                SmoothingKernel::Difference => (b_avg[i] - f_avg[i]) / 2.0,
            });
        }
        output
    }
}

// END: F32ExponentialSmoothingTransformer

// BEGIN: F32NormalizeByAverageTransformer

pub struct F32NormalizeByAverageTransformer {}

impl F32NormalizeByAverageTransformer {
    pub fn new() -> Self {
        Self {}
    }
}

impl Transformer for F32NormalizeByAverageTransformer {
    type Input = Vec<f32>;
    type Output = Vec<f32>;

    fn transform(&mut self, input: Vec<f32>) -> Vec<f32> {
        let input_len = input.len();
        assert!(input_len > 0, "input length must be greater than 0");
        let average = input.iter().fold(0.0, |acc, &x| acc + x) / input_len as f32;
        input.iter().map(|&v| v - average).collect()
    }
}

// END: F32NormalizeByAverageTransformer

const F32_LOG_C: f32 = 1e-7;

// BEGIN: F32NormalizeByLogTransformer

pub struct F32NormalizeByLogTransformer {}

impl F32NormalizeByLogTransformer {
    pub fn new() -> Self {
        Self {}
    }
}

impl Transformer for F32NormalizeByLogTransformer {
    type Input = Vec<f32>;
    type Output = Vec<f32>;

    fn transform(&mut self, input: Vec<f32>) -> Vec<f32> {
        input.iter().map(|&v| (v + F32_LOG_C).ln()).collect()
    }
}

// END: F32NormalizeByLogTransformer

// BEGIN: F32HannWindowTransformer

pub struct F32HannWindowTransformer {}

impl F32HannWindowTransformer {
    pub fn new() -> Self {
        Self {}
    }
}

impl Transformer for F32HannWindowTransformer {
    type Input = Vec<f32>;
    type Output = Vec<f32>;

    fn transform(&mut self, input: Vec<f32>) -> Vec<f32> {
        let start = SystemTime::now();
        let input_len = input.len();
        assert!(input_len > 0, "input length must be greater than 0");
        let hor_scale = std::f32::consts::PI / (input_len as f32);
        let out = input
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as f32 * hor_scale).sin().powi(2) * v)
            .collect();
        println!("Hann({})", start.elapsed().unwrap().as_micros());
        out
    }
}

// END: F32HannWindowTransformer

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

// BEGIN: F32DivideTransformer

pub struct F32DivideTransformer {
    by: f32,
}

impl F32DivideTransformer {
    pub fn new(by: f32) -> Self {
        assert!(by != 0.0, "cannot divide by 0");
        Self { by }
    }
}

impl Transformer for F32DivideTransformer {
    type Input = f32;
    type Output = f32;

    fn transform(&mut self, input: f32) -> f32 {
        input / self.by
    }
}

// END: F32DivideTransformer

// BEGIN: VectorTransformer

pub struct VectorTransformer<I, O: Clone> {
    transformer: Box<dyn Transformer<Input = I, Output = O>>,
}

impl<I: Clone, O: Copy> VectorTransformer<I, O> {
    pub fn new(transformer: Box<dyn Transformer<Input = I, Output = O>>) -> Self {
        Self { transformer }
    }
}

impl<I: Clone, O: Copy> Transformer for VectorTransformer<I, O> {
    type Input = Vec<I>;
    type Output = Vec<O>;

    fn transform(&mut self, input: Vec<I>) -> Vec<O> {
        let start = SystemTime::now();
        let out = input
            .iter()
            .map(|v| self.transformer.transform(v.clone()))
            .collect();
        println!("VectorTime({})", start.elapsed().unwrap().as_micros());
        out
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

// BEGIN: VectorCacheTransformer

pub struct VectorCacheTransformer<I: Copy, O: Clone> {
    transformers: Vec<Box<dyn Transformer<Input = I, Output = Option<O>>>>,
    cache: Vec<O>,
}

impl<I: Copy, O: Clone> VectorCacheTransformer<I, O> {
    pub fn new(
        transformers: Vec<Box<dyn Transformer<Input = I, Output = Option<O>>>>,
        default_value: O,
    ) -> Self {
        let mut cache = Vec::with_capacity(transformers.len());
        for _ in 0..transformers.len() {
            cache.push(default_value.clone());
        }
        Self {
            transformers,
            cache,
        }
    }
}

impl<I: Copy, O: Clone> Transformer for VectorCacheTransformer<I, O> {
    type Input = Vec<I>;
    type Output = Option<Vec<O>>;

    fn transform(&mut self, input: Vec<I>) -> Option<Vec<O>> {
        assert!(
            input.len() == self.transformers.len(),
            "Input length must be same as length of tranformers"
        );
        let mut has_update = false;
        let mut idx = 0;
        for (input_val, transformer) in input.iter().zip(self.transformers.iter_mut()) {
            match transformer.transform(*input_val) {
                Some(output_val) => {
                    has_update = true;
                    self.cache[idx] = output_val;
                }
                None => {}
            }
            idx += 1;
        }

        if has_update {
            Some(self.cache.clone())
        } else {
            None
        }
    }
}

// END: VectorCacheTransformer

// BEGIN: F32WindowAverageTransformer

pub struct F32WindowAverageTransformer {
    window_size: usize,
}

impl F32WindowAverageTransformer {
    pub fn new(window_size: usize) -> Self {
        assert!(window_size > 0, "window_size must be greater than 0");
        Self { window_size }
    }
}

impl Transformer for F32WindowAverageTransformer {
    type Input = Vec<f32>;
    type Output = Vec<f32>;

    fn transform(&mut self, input: Vec<f32>) -> Vec<f32> {
        let input_len = input.len();
        assert!(
            input_len % self.window_size == 0,
            "Length of input must be divisible by window_size"
        );
        let num_windows = input_len / self.window_size;
        let mut output = Vec::with_capacity(num_windows);
        for window_idx in 0..num_windows {
            let mut total = 0.0;
            for offset in 0..self.window_size {
                total += input[self.window_size * window_idx + offset];
            }
            output.push(total / (self.window_size as f32));
        }
        output
    }
}

// END: F32WindowAverageTransformer

// BEGIN: F32AverageTransformer

pub struct F32AverageTransformer {}

impl F32AverageTransformer {
    pub fn new() -> Self {
        Self {}
    }
}

impl Transformer for F32AverageTransformer {
    type Input = Vec<f32>;
    type Output = f32;

    fn transform(&mut self, input: Vec<f32>) -> f32 {
        assert!(input.len() > 0, "input must be non-empty");
        let mut total = 0.0;
        for &input_val in input.iter() {
            total += input_val;
        }
        return total / (input.len() as f32);
    }
}

// END: F32AverageTransformer

// BEGIN: VectorSubTransformer

pub struct VectorSubTransformer<T: Copy> {
    start: usize,
    end: usize,
    phantom: PhantomData<T>,
}

impl<T: Copy> VectorSubTransformer<T> {
    pub fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            phantom: PhantomData,
        }
    }
}

impl<T: Copy> Transformer for VectorSubTransformer<T> {
    type Input = Vec<T>;
    type Output = Vec<T>;

    fn transform(&mut self, input: Vec<T>) -> Vec<T> {
        input[self.start..self.end].to_vec()
    }
}

// END: VectorSubTransformer

// BEGIN: ZipTransformer

pub struct ZipTransformer<T: Copy> {
    phantom: PhantomData<T>,
}

impl<T: Copy> ZipTransformer<T> {
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<T: Copy> Transformer for ZipTransformer<T> {
    type Input = Vec<Vec<T>>;
    type Output = Vec<Vec<T>>;

    fn transform(&mut self, input: Vec<Vec<T>>) -> Vec<Vec<T>> {
        let n = input.len();
        if n == 0 {
            return input;
        }
        let m = input[0].len();

        let mut output = Vec::with_capacity(m);
        for _ in 0..m {
            output.push(Vec::with_capacity(n));
        }

        for i in 0..n {
            assert!(input[i].len() == m, "Mismatched length for row");
            for j in 0..m {
                output[j].push(input[i][j]);
            }
        }

        output
    }
}

// END: ZipTransformer
