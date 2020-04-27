use crate::config::SpectrogramConfig;
use crate::mappers::{f32_hann_window, f32_log_compression};
use crate::transforms::{
    FFTtransformer, IteratorCollector, IteratorEnumMapper, IteratorMapper, IteratorSubSequencer,
    OptionalPipelineTransformer, StutterAggregatorTranformer, Transformer, TwoChannel,
    VectorTwoChannelCombiner,
};

fn build_single_channel_transformer(
    config: &SpectrogramConfig,
) -> Box<dyn Transformer<Input = i16, Output = Option<Box<dyn Iterator<Item = f32>>>>> {
    let stutter_transformer =
        StutterAggregatorTranformer::<i16>::new(config.buffer_size, config.stutter_size);
    let fft_transformer = FFTtransformer::<f32>::new(config.buffer_size);
    let buffer_size = config.buffer_size;
    let hann_function = move |(i, v): (usize, f32)| f32_hann_window(i, v, buffer_size);

    let casted_to_f32 = OptionalPipelineTransformer::new(
        Box::new(stutter_transformer),
        Box::new(IteratorMapper::new(|i: i16| i as f32)),
    );
    let normalized = OptionalPipelineTransformer::new(
        Box::new(casted_to_f32),
        Box::new(IteratorMapper::new(|f: f32| f / i16::max_value() as f32)),
    );
    let hanned = OptionalPipelineTransformer::new(
        Box::new(normalized),
        Box::new(IteratorEnumMapper::new(hann_function)),
    );
    Box::new(OptionalPipelineTransformer::new(
        Box::new(hanned),
        Box::new(fft_transformer),
    ))
}

pub fn build_spectrogram_transformer(
    config: &SpectrogramConfig,
    channel_size: usize,
) -> Box<dyn Transformer<Input = TwoChannel<i16>, Output = Option<Vec<f32>>>> {
    assert!(channel_size == 2, "Only two channels currently supported");

    let combined_channels = VectorTwoChannelCombiner::new(
        build_single_channel_transformer(config),
        build_single_channel_transformer(config),
    );

    let averaged_channels = OptionalPipelineTransformer::new(
        Box::new(combined_channels),
        Box::new(IteratorMapper::new(|x: TwoChannel<f32>| (x.0 + x.1) / 2.0)),
    );

    let maybe_log_normalized: Box<
        dyn Transformer<Input = TwoChannel<i16>, Output = Option<Box<dyn Iterator<Item = f32>>>>,
    > = if config.log_scaling {
        Box::new(OptionalPipelineTransformer::new(
            Box::new(averaged_channels),
            Box::new(IteratorMapper::new(f32_log_compression)),
        ))
    } else {
        Box::new(averaged_channels)
    };

    let maybe_sub_sequenced: Box<
        dyn Transformer<Input = TwoChannel<i16>, Output = Option<Box<dyn Iterator<Item = f32>>>>,
    > = match config.band_subset {
        Some(subset) => Box::new(OptionalPipelineTransformer::new(
            maybe_log_normalized,
            Box::new(IteratorSubSequencer::new(subset.start, subset.end)),
        )),
        None => maybe_log_normalized,
    };

    Box::new(OptionalPipelineTransformer::new(
        maybe_sub_sequenced,
        Box::new(IteratorCollector::new()),
    ))
}
