use std::cell::RefCell;
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use num_complex::Complex;
use num_traits::Zero;
use rustfft::{FFTplanner, FFT};

use crate::cyclical_buffer::CyclicalBuffer;
use crate::spectrogram::TwoChannel;

pub trait Transformer {
    type Input;
    type Output;

    fn transform(&mut self, input: Self::Input) -> Self::Output;
}

// BEGIN: FFTTransformer

pub struct FFTTransformer {
    fft: Arc<dyn FFT<f32>>,
}

impl FFTTransformer {
    pub fn new(size: usize) -> Self {
        Self {
            fft: FFTplanner::new(false).plan_fft(size),
        }
    }
}

impl Transformer for FFTTransformer {
    type Input = Box<dyn Iterator<Item = f32>>;
    type Output = Box<dyn Iterator<Item = Complex<f32>>>;

    fn transform(
        &mut self,
        input: Box<dyn Iterator<Item = f32>>,
    ) -> Box<dyn Iterator<Item = Complex<f32>>> {
        let mut m_input: Vec<_> = input.map(|v| Complex::new(v, 0.0)).collect();
        let mut m_output = vec![Complex::zero(); self.fft.len()];
        self.fft.process(&mut m_input, &mut m_output);

        // normalize
        let len_sqrt = (self.fft.len() as f32).sqrt();
        Box::new(m_output.into_iter().map(move |elem| elem / len_sqrt))
    }
}

// END: FFTTransformer

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

// BEGIN: PipelineTransformer

pub struct PipelineTransformer<I, M, O> {
    transformer_one: Box<dyn Transformer<Input = I, Output = M>>,
    transformer_two: Box<dyn Transformer<Input = M, Output = O>>,
}

impl<I, M, O> PipelineTransformer<I, M, O> {
    pub fn new(
        transformer_one: Box<dyn Transformer<Input = I, Output = M>>,
        transformer_two: Box<dyn Transformer<Input = M, Output = O>>,
    ) -> Self {
        Self {
            transformer_one,
            transformer_two,
        }
    }
}

impl<I, M, O> Transformer for PipelineTransformer<I, M, O> {
    type Input = I;
    type Output = O;

    fn transform(&mut self, input: I) -> O {
        self.transformer_two
            .transform(self.transformer_one.transform(input))
    }
}

// END: PipelineTransformer

// BEGIN: Mapper

pub struct Mapper<I, O, F>
where
    F: Fn(I) -> O + Copy + 'static,
{
    f: F,
    i_phantom: PhantomData<I>,
    o_phantom: PhantomData<O>,
}

impl<I, O, F: Fn(I) -> O + Copy + 'static> Mapper<I, O, F> {
    pub fn new(f: F) -> Self {
        Self {
            f,
            i_phantom: PhantomData,
            o_phantom: PhantomData,
        }
    }
}

impl<I: 'static, O, F: Fn(I) -> O + Copy + 'static> Transformer for Mapper<I, O, F> {
    type Input = I;
    type Output = O;

    fn transform(&mut self, input: I) -> O {
        (self.f)(input)
    }
}

// END: Mapper

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
        input.collect()
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

// BEGIN: BRPeakPicker
// Peak picker as defined in the BeatRoot paper:
// https://www.eecs.qmul.ac.uk/~simond/pub/2007/jnmr07.pdf

// Computed over one kiss l2 normalized. Should recompute over larger number of songs and l1
// normalization.
const BR_MEAN: f32 = 2.238;
const BR_STDEV: f32 = 1.705;

pub struct BRPeakPicker {
    buf: CyclicalBuffer<f32>,
    cur_idx: u64,
    decay_avg: f32,
    w: usize,
    m: usize,
    alpha: f32,
    gamma: f32,
}

impl BRPeakPicker {
    pub fn new(w: usize, m: usize, gamma: f32, alpha: f32) -> Self {
        Self {
            buf: CyclicalBuffer::new(w * (m + 1) + 1),
            cur_idx: 0,
            decay_avg: 0.0,
            w,
            m,
            alpha,
            gamma,
        }
    }
}

impl Transformer for BRPeakPicker {
    type Input = f32;
    type Output = Option<(u64, f32)>; // Index of picked peak.

    fn transform(&mut self, input: f32) -> Option<(u64, f32)> {
        self.cur_idx += 1;
        self.buf.push((input - BR_MEAN) / BR_STDEV);

        if self.buf.full() {
            let values = self.buf.get_values();
            let value = values[values.len() - self.w - 1];
            let cond_one = value
                >= *values
                    .iter()
                    .skip(self.w * (self.m - 1))
                    .max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
                    .unwrap();
            let cond_two = value >= (values.iter().sum::<f32>() / values.len() as f32) + self.gamma;

            let cond_three = value >= self.decay_avg;
            self.decay_avg = value.max(self.alpha * self.decay_avg + (1.0 - self.alpha) * value);

            if cond_one & cond_two & cond_three {
                Some((self.cur_idx - self.w as u64, value))
            } else {
                None
            }
        } else {
            None
        }
    }
}

// END: BRPeakPicker

// BEGIN: BRClusterer

#[derive(Debug, PartialEq)]
struct BRCluster {
    pub latest_onset: f32,
    pub diff: f32,
    pub value: f32,
    pub size: usize,
}

impl BRCluster {
    pub fn new(onset: f32, diff: f32, value: f32) -> Self {
        Self {
            latest_onset: onset,
            diff,
            value,
            size: 1,
        }
    }

    pub fn update(&mut self, onset: f32, diff: f32, value: f32, decay: f32) {
        self.latest_onset = onset;
        self.size += 1;
        self.diff = self.diff + (diff - self.diff) / (self.size as f32);
        self.value = self.value * decay + value * (1.0 - decay);
    }

    fn score(&self) -> f32 {
        return self.value * self.size as f32 * self.diff;
    }
}

pub struct BRClusterer {
    cur_idx: u64,
    min_diff: f32,
    max_diff: f32,
    threshold: f32,
    decay: f32,
    clusters: Vec<(f32, Vec<Rc<RefCell<BRCluster>>>)>, // (onset, clusters ending on onset)
    best_cluster: Option<Rc<RefCell<BRCluster>>>,
}

impl BRClusterer {
    pub fn new(min_diff: f32, max_diff: f32, threshold: f32, decay: f32) -> Self {
        Self {
            cur_idx: 0,
            min_diff,
            max_diff,
            threshold,
            decay,
            clusters: Vec::new(),
            best_cluster: None,
        }
    }
}

impl Transformer for BRClusterer {
    type Input = Option<(u64, f32)>;
    type Output = bool;

    fn transform(&mut self, input: Option<(u64, f32)>) -> bool {
        self.cur_idx += 1;

        match input {
            Some((onset, value)) => {
                let onset_fl = onset as f32;
                let mut clusters = Vec::new();
                let mut cur_onset_clusters = Vec::new();
                for (other_onset, other_clusters) in &self.clusters {
                    let diff = onset_fl - *other_onset;
                    if diff > self.max_diff {
                        let mut clear_best = false;
                        match &self.best_cluster {
                            Some(best_cluster) => {
                                for cluster in other_clusters {
                                    if *cluster.borrow() == *best_cluster.borrow() {
                                        clear_best = true;
                                    }
                                }
                            }
                            _ => {}
                        }
                        if clear_best {
                            self.best_cluster = None;
                        }
                        continue;
                    }

                    if diff < self.min_diff {
                        clusters.push((*other_onset, other_clusters.clone()));
                    } else {
                        // Collect clusters that are being extended by the current offset
                        let mut new_clusters = Vec::new();
                        let new_other_clusters: Vec<Rc<RefCell<BRCluster>>> = other_clusters
                            .iter()
                            .filter(|&cluster| {
                                let cluster_diff = cluster.borrow().diff;
                                let mut keep = true;
                                for mult in &[1.0, 2.0, 3.0] {
                                    let adj_diff = diff / mult;
                                    if (cluster_diff - adj_diff).abs() <= self.threshold {
                                        cluster
                                            .borrow_mut()
                                            .update(onset_fl, adj_diff, value, self.decay);
                                        new_clusters.push(Rc::clone(cluster));
                                        keep = false;
                                        break;
                                    }
                                }
                                keep
                            })
                            .map(|x| Rc::clone(x))
                            .collect();
                        clusters.push((*other_onset, new_other_clusters));

                        if new_clusters.len() > 0 {
                            for cluster in new_clusters {
                                cur_onset_clusters.push(Rc::clone(&cluster));
                            }
                        } else {
                            cur_onset_clusters
                                .push(Rc::new(RefCell::new(BRCluster::new(onset_fl, diff, value))));
                        }
                    }
                }

                // Update best cluster if it changes.
                let mut highest_score = match &self.best_cluster {
                    Some(cluster_cell) => cluster_cell.borrow().score(),
                    None => 0.0,
                };
                for cluster_cell in &mut cur_onset_clusters {
                    let score = cluster_cell.borrow().score();
                    if score > highest_score {
                        highest_score = score;
                        self.best_cluster = Some(Rc::clone(cluster_cell));
                    }
                }

                clusters.push((onset_fl, cur_onset_clusters));
                self.clusters = clusters;
            }
            None => {}
        }

        match &self.best_cluster {
            Some(cluster_cell) => {
                let cluster = cluster_cell.borrow();
                if (self.cur_idx as i64 - cluster.latest_onset as i64) % cluster.diff as i64 == 0 {
                    println!("Best\n{:?} with score {}", cluster, cluster.score());
                    true
                } else {
                    false
                }
            }
            None => false,
        }
    }
}

// END: BRClusterer

// UNUSED...

// BEGIN: VectorTwoChannelCombiner

#[allow(dead_code)]
pub struct VectorTwoChannelCombiner<I, O: Clone> {
    transformer_one: Box<dyn Transformer<Input = I, Output = Option<Box<dyn Iterator<Item = O>>>>>,
    transformer_two: Box<dyn Transformer<Input = I, Output = Option<Box<dyn Iterator<Item = O>>>>>,
}

#[allow(dead_code)]
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

#[allow(dead_code)]
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
