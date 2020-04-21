use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;

use spectrum_audio::build_spectrogram_transformer;
use spectrum_audio::config::{SpectrogramConfig, SubsetConfig};

const NUM_SAMPLES: usize = 44100;

pub fn criterion_benchmark(c: &mut Criterion) {
    let config = SpectrogramConfig {
        buffer_size: 1024,
        stutter_size: 512,
        band_subset: Some(SubsetConfig { start: 0, end: 512 }),
        log_scaling: false,
    };
    let mut transformer = build_spectrogram_transformer(&config, 2);
    c.bench_function("spectrogram transformer", |b| {
        b.iter(|| {
            let mut rng = rand::thread_rng();
            for _ in 0..NUM_SAMPLES {
                transformer.transform(black_box((rng.gen(), rng.gen())));
            }
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
