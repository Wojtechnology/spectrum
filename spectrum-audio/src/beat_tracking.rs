use num_complex::Complex;

use crate::config::BeatTrackingConfig;
use crate::math::{complex_l1_norm, spectral_flux};
use crate::transforms::{
    BRClusterer, BRPeakPicker, IteratorMapper, Mapper, PipelineTransformer, Transformer,
};

pub fn beat_tracking_transformer(
    config: &BeatTrackingConfig,
) -> Box<dyn Transformer<Input = Box<dyn Iterator<Item = Complex<f32>>>, Output = Option<f32>>> {
    let l1_transformer = IteratorMapper::new(complex_l1_norm);
    let spectral_fluxed = PipelineTransformer::new(
        Box::new(l1_transformer),
        Box::new(Mapper::new(spectral_flux)),
    );
    let peak_picked = PipelineTransformer::new(
        Box::new(spectral_fluxed),
        Box::new(BRPeakPicker::new(
            config.w,
            config.m,
            config.gamma,
            config.alpha,
        )),
    );
    Box::new(PipelineTransformer::new(
        Box::new(peak_picked),
        Box::new(BRClusterer::new(
            config.min_diff,
            config.max_diff,
            config.threshold,
            config.decay,
        )),
    ))
}
