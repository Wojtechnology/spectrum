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
    type Input = Box<dyn Iterator<Item = T>>;
    type Output = Box<dyn Iterator<Item = T>>;

    fn transform(&mut self, input: Box<dyn Iterator<Item = T>>) -> Box<dyn Iterator<Item = T>> {
        let start = SystemTime::now();
        let mut m_input: Vec<_> = input.map(|v| T::real_to_complex(v)).collect();
        let mut m_output = vec![Complex::zero(); self.fft.len()];
        self.fft.process(&mut m_input, &mut m_output);

        // normalize
        let len_sqrt = T::cast_and_sqrt(self.fft.len());
        let out = m_output.into_iter().map(move |elem| {
            let c = elem / len_sqrt;
            T::sqrt(c.re * c.re + c.im * c.im)
        });
        println!("FFTTime({})", start.elapsed().unwrap().as_micros());
        Box::new(out)
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

impl<T: Copy + 'static> Transformer for StutterAggregatorTranformer<T> {
    type Input = T;
    type Output = Option<Box<dyn Iterator<Item = T>>>;

    fn transform(&mut self, input: T) -> Option<Box<dyn Iterator<Item = T>>> {
        self.buf.push(input);
        if self.starting_cursor < self.buffer_size {
            self.starting_cursor += 1;
            None
        } else {
            let cur_stutter_cursor = self.stutter_cursor;
            self.stutter_cursor = (self.stutter_cursor + 1) % self.stutter_size;
            if cur_stutter_cursor == 0 {
                Some(Box::new(self.buf.get_values().into_iter()))
            } else {
                None
            }
        }
    }
}

// END: StutterAggregatorTranformer

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

// BEGIN: VectorTwoChannelCombiner

pub type TwoChannel<T> = (T, T);

pub struct VectorTwoChannelCombiner<I, O: Clone> {
    transformer_one: Box<dyn Transformer<Input = I, Output = Option<Box<dyn Iterator<Item = O>>>>>,
    transformer_two: Box<dyn Transformer<Input = I, Output = Option<Box<dyn Iterator<Item = O>>>>>,
}

impl<I, O: Clone + 'static> VectorTwoChannelCombiner<I, O> {
    pub fn new(
        transformer_one: Box<
            dyn Transformer<Input = I, Output = Option<Box<dyn Iterator<Item = O>>>>,
        >,
        transformer_two: Box<
            dyn Transformer<Input = I, Output = Option<Box<dyn Iterator<Item = O>>>>,
        >,
    ) -> Self {
        Self {
            transformer_one,
            transformer_two,
        }
    }
}

impl<I, O: Clone + 'static> Transformer for VectorTwoChannelCombiner<I, O> {
    type Input = TwoChannel<I>;
    type Output = Option<Box<dyn Iterator<Item = TwoChannel<O>>>>;

    fn transform(
        &mut self,
        input: TwoChannel<I>,
    ) -> Option<Box<dyn Iterator<Item = TwoChannel<O>>>> {
        match (
            self.transformer_one.transform(input.0),
            self.transformer_two.transform(input.1),
        ) {
            (Some(l), Some(r)) => Some(Box::new(l.zip(r).map(|(l, r)| (l, r)))),
            (None, None) => None,
            // Possibly just log and don't actually panic, although this should really never
            // happen unless something is set up wrong with the program.
            _ => panic!("Unsynchronized channel transformers"),
        }
    }
}

// END: VectorTwoChannelCombiner

// BEGIN: IteratorMapper

pub struct IteratorMapper<I, O, F>
where
    F: Fn(I) -> O + Copy + 'static,
{
    f: F,
    i_phantom: PhantomData<I>,
    o_phantom: PhantomData<O>,
}

impl<I, O, F: Fn(I) -> O + Copy + 'static> IteratorMapper<I, O, F> {
    pub fn new(f: F) -> Self {
        Self {
            f,
            i_phantom: PhantomData,
            o_phantom: PhantomData,
        }
    }
}

impl<I: 'static, O, F: Fn(I) -> O + Copy + 'static> Transformer for IteratorMapper<I, O, F> {
    type Input = Box<dyn Iterator<Item = I>>;
    type Output = Box<dyn Iterator<Item = O>>;

    fn transform(&mut self, input: Box<dyn Iterator<Item = I>>) -> Box<dyn Iterator<Item = O>> {
        Box::new(input.map(self.f))
    }
}

// END: IteratorMapper

// BEGIN: IteratorEnumMapper

pub struct IteratorEnumMapper<I, O, F>
where
    F: Fn((usize, I)) -> O + Copy + 'static,
{
    f: F,
    i_phantom: PhantomData<I>,
    o_phantom: PhantomData<O>,
}

impl<I, O, F: Fn((usize, I)) -> O + Copy + 'static> IteratorEnumMapper<I, O, F> {
    pub fn new(f: F) -> Self {
        Self {
            f,
            i_phantom: PhantomData,
            o_phantom: PhantomData,
        }
    }
}

impl<I: 'static, O, F: Fn((usize, I)) -> O + Copy + 'static> Transformer
    for IteratorEnumMapper<I, O, F>
{
    type Input = Box<dyn Iterator<Item = I>>;
    type Output = Box<dyn Iterator<Item = O>>;

    fn transform(&mut self, input: Box<dyn Iterator<Item = I>>) -> Box<dyn Iterator<Item = O>> {
        Box::new(input.enumerate().map(self.f))
    }
}

// END: IteratorEnumMapper

// BEGIN: IteratorCollector

pub struct IteratorCollector<T> {
    phantom: PhantomData<T>,
}

impl<T> IteratorCollector<T> {
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<T> Transformer for IteratorCollector<T> {
    type Input = Box<dyn Iterator<Item = T>>;
    type Output = Vec<T>;

    fn transform(&mut self, input: Box<dyn Iterator<Item = T>>) -> Vec<T> {
        let start = SystemTime::now();
        let out = input.collect();
        println!("CollectTime({})", start.elapsed().unwrap().as_micros());
        out
    }
}

// END: IteratorCollector

// BEGIN: IteratorSubSequencer

pub struct IteratorSubSequencer<T> {
    start: usize,
    end: usize,
    phantom: PhantomData<T>,
}

impl<T> IteratorSubSequencer<T> {
    pub fn new(start: usize, end: usize) -> Self {
        assert!(start <= end, "end must be at least start");
        Self {
            start,
            end,
            phantom: PhantomData,
        }
    }
}

impl<T: 'static> Transformer for IteratorSubSequencer<T> {
    type Input = Box<dyn Iterator<Item = T>>;
    type Output = Box<dyn Iterator<Item = T>>;

    fn transform(&mut self, input: Box<dyn Iterator<Item = T>>) -> Box<dyn Iterator<Item = T>> {
        Box::new(input.skip(self.start).take(self.end - self.start))
    }
}

// END: IteratorSubSequencer
