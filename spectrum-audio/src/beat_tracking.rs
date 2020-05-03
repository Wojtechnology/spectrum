use num_complex::Complex;

use crate::math::{complex_l1_norm, spectral_flux};
use crate::transforms::{IteratorMapper, Mapper, PipelineTransformer, Transformer};

fn beat_tracking_transformer(
) -> Box<dyn Transformer<Input = Box<dyn Iterator<Item = Complex<f32>>>, Output = f32>> {
    let l1_transformer = IteratorMapper::new(complex_l1_norm);
    Box::new(PipelineTransformer::new(
        Box::new(l1_transformer),
        Box::new(Mapper::new(spectral_flux)),
    ))
}
