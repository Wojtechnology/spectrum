use num_complex::Complex;

use crate::config::SpectrogramConfig;
use crate::math::{complex_l2_norm, hann_window, log_compression};
use crate::transforms::{
    FFTTransformer, IteratorCollector, IteratorEnumMapper, IteratorMapper, IteratorSubSequencer,
    Mapper, OptionalPipelineTransformer, PipelineTransformer, StutterAggregatorTranformer,
    Transformer, TwoChannel,
};

#[inline(always)]
fn normalize_i16(v: i16) -> f32 {
    return v as f32 / i16::max_value() as f32;
}

fn iter_fft_transformer(
    config: &SpectrogramConfig,
    channel_size: usize,
) -> Box<
    dyn Transformer<
        Input = TwoChannel<i16>,
        Output = Option<Box<dyn Iterator<Item = Complex<f32>>>>,
    >,
> {
    assert!(channel_size == 2, "Only two channels currently supported");

    let buffer_size = config.buffer_size;
    let hann_function = move |(i, v): (usize, f32)| hann_window(i, v, buffer_size);

    let combined_channels =
        Mapper::new(|(l, r): TwoChannel<i16>| (normalize_i16(l) + normalize_i16(r)) / 2.0);
    let stutter = StutterAggregatorTranformer::<f32>::new(config.buffer_size, config.stutter_size);

    let stuttered_window = PipelineTransformer::new(Box::new(combined_channels), Box::new(stutter));
    let hanned_window = OptionalPipelineTransformer::new(
        Box::new(stuttered_window),
        Box::new(IteratorEnumMapper::new(hann_function)),
    );
    Box::new(OptionalPipelineTransformer::new(
        Box::new(hanned_window),
        Box::new(FFTTransformer::new(config.buffer_size)),
    ))
}

pub fn fft_transformer(
    config: &SpectrogramConfig,
    channel_size: usize,
) -> Box<dyn Transformer<Input = TwoChannel<i16>, Output = Option<Vec<Complex<f32>>>>> {
    Box::new(OptionalPipelineTransformer::new(
        iter_fft_transformer(config, channel_size),
        Box::new(IteratorCollector::new()),
    ))
}

fn fft_viz_transformer(
    config: &SpectrogramConfig,
) -> Box<dyn Transformer<Input = Box<dyn Iterator<Item = Complex<f32>>>, Output = Vec<f32>>> {
    let l2_transformer = IteratorMapper::new(complex_l2_norm);
    let maybe_sub_sequenced: Box<
        dyn Transformer<
            Input = Box<dyn Iterator<Item = Complex<f32>>>,
            Output = Box<dyn Iterator<Item = f32>>,
        >,
    > = match config.band_subset {
        Some(subset) => Box::new(PipelineTransformer::new(
            Box::new(l2_transformer),
            Box::new(IteratorSubSequencer::new(subset.start, subset.end)),
        )),
        None => Box::new(l2_transformer),
    };

    let maybe_log_normalized: Box<
        dyn Transformer<
            Input = Box<dyn Iterator<Item = Complex<f32>>>,
            Output = Box<dyn Iterator<Item = f32>>,
        >,
    > = if config.log_scaling {
        Box::new(PipelineTransformer::new(
            maybe_sub_sequenced,
            Box::new(IteratorMapper::new(log_compression)),
        ))
    } else {
        maybe_sub_sequenced
    };
    Box::new(PipelineTransformer::new(
        maybe_log_normalized,
        Box::new(IteratorCollector::new()),
    ))
}

pub fn spectrogram_viz_transformer(
    config: &SpectrogramConfig,
    channel_size: usize,
) -> Box<dyn Transformer<Input = TwoChannel<i16>, Output = Option<Vec<f32>>>> {
    let fft_transformer = iter_fft_transformer(config, channel_size);
    let viz_transformer = fft_viz_transformer(config);
    Box::new(OptionalPipelineTransformer::new(
        fft_transformer,
        viz_transformer,
    ))
}
