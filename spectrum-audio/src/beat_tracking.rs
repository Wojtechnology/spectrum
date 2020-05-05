use num_complex::Complex;

use crate::math::{complex_l1_norm, spectral_flux};
use crate::transforms::{
    BRClusterer, BRPeakPicker, IteratorMapper, Mapper, PipelineTransformer, Transformer,
};

const GAMMA: f32 = 0.35;
const ALPHA: f32 = 0.84;

pub fn beat_tracking_transformer(
) -> Box<dyn Transformer<Input = Box<dyn Iterator<Item = Complex<f32>>>, Output = bool>> {
    let l1_transformer = IteratorMapper::new(complex_l1_norm);
    let spectral_fluxed = PipelineTransformer::new(
        Box::new(l1_transformer),
        Box::new(Mapper::new(spectral_flux)),
    );
    let peak_picked = PipelineTransformer::new(
        Box::new(spectral_fluxed),
        Box::new(BRPeakPicker::new(10, 3, GAMMA, ALPHA)),
    );
    Box::new(PipelineTransformer::new(
        Box::new(peak_picked),
        Box::new(BRClusterer::new(60, 600, 5)),
    ))
}
