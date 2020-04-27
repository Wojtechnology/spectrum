use crate::config::SpectrogramConfig;
use crate::mappers::{f32_hann_window, f32_log_compression};
use crate::transforms::{
    FFTTransformer, IteratorCollector, IteratorEnumMapper, IteratorMapper, IteratorSubSequencer,
    Mapper, OptionalPipelineTransformer, PipelineTransformer, StutterAggregatorTranformer,
    Transformer, TwoChannel, VectorTwoChannelCombiner,
};

#[inline(always)]
fn normalize_i16(v: i16) -> f32 {
    return v as f32 / i16::max_value() as f32;
}

pub fn build_spectrogram_transformer(
    config: &SpectrogramConfig,
    channel_size: usize,
) -> Box<dyn Transformer<Input = TwoChannel<i16>, Output = Option<Vec<f32>>>> {
    assert!(channel_size == 2, "Only two channels currently supported");

    let buffer_size = config.buffer_size;
    let hann_function = move |(i, v): (usize, f32)| f32_hann_window(i, v, buffer_size);

    let combined_channels =
        Mapper::new(|(l, r): TwoChannel<i16>| (normalize_i16(l) + normalize_i16(r)) / 2.0);
    let stutter = StutterAggregatorTranformer::<f32>::new(config.buffer_size, config.stutter_size);

    let stuttered_window = PipelineTransformer::new(Box::new(combined_channels), Box::new(stutter));
    let hanned_window = OptionalPipelineTransformer::new(
        Box::new(stuttered_window),
        Box::new(IteratorEnumMapper::new(hann_function)),
    );
    let ffted = OptionalPipelineTransformer::new(
        Box::new(hanned_window),
        Box::new(FFTTransformer::<f32>::new(config.buffer_size)),
    );

    let maybe_sub_sequenced: Box<
        dyn Transformer<Input = TwoChannel<i16>, Output = Option<Box<dyn Iterator<Item = f32>>>>,
    > = match config.band_subset {
        Some(subset) => Box::new(OptionalPipelineTransformer::new(
            Box::new(ffted),
            Box::new(IteratorSubSequencer::new(subset.start, subset.end)),
        )),
        None => Box::new(ffted),
    };

    let maybe_log_normalized: Box<
        dyn Transformer<Input = TwoChannel<i16>, Output = Option<Box<dyn Iterator<Item = f32>>>>,
    > = if config.log_scaling {
        Box::new(OptionalPipelineTransformer::new(
            maybe_sub_sequenced,
            Box::new(IteratorMapper::new(f32_log_compression)),
        ))
    } else {
        maybe_sub_sequenced
    };

    Box::new(OptionalPipelineTransformer::new(
        maybe_log_normalized,
        Box::new(IteratorCollector::new()),
    ))
}
