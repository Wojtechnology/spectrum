use std::io::Read;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_yaml;

pub trait ValidatedConfig {
    fn validate(&self) -> Result<(), String>;
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct SubsetConfig {
    pub start: usize,
    pub end: usize,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct SpectrogramConfig {
    // Size of circular buffer for audio data points and as a result, number of data points that
    // are input into the FFT transform. The output of the transform will be of the same size but
    // the number of usable output datapoints are only the values for positive coefficients, which
    // means that we can only use `buffer_size / 2`.
    pub buffer_size: usize,
    // Every `stutter_size` data points, we run the FFT transform on the current circular buffer
    // (assuming there are currently enough values in it).
    pub stutter_size: usize,
    // What subset of bands to include in the output. Usually you want this to be buffer_size / 2
    // so that you pick up only the positive coefficients.
    pub band_subset: Option<SubsetConfig>,
    // Scale the output of the FFT logarithmically.
    pub log_scaling: bool,
}

fn assertion(cond: bool, err_msg: &str) -> Result<(), String> {
    if cond {
        Ok(())
    } else {
        Err(err_msg.to_string())
    }
}

impl ValidatedConfig for SpectrogramConfig {
    fn validate(&self) -> Result<(), String> {
        assertion(self.buffer_size > 0, "buffer_size must be greater than 0")?;
        assertion(self.stutter_size > 0, "stutter_size must be greater than 0")?;
        assertion(
            self.buffer_size % 2 == 0,
            "buffer_size must be divisible by 2",
        )?;
        match &self.band_subset {
            Some(subset) => {
                assertion(
                    subset.start <= subset.end,
                    "band_subset.start must be at most band_subset.end",
                )?;
                Result::<(), String>::Ok(())
            }
            None => Ok(()),
        }?;
        Ok(())
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct BeatTrackingConfig {
    // Hyperparameter for peak picking average decay.
    pub alpha: f32,
    // Parameters for window size used for finding the peak:
    // Cond 1: true iff current point is the largest in [-w, w] window.
    // Cond 2: true iff average of values in [-m * w, w] window + gamma is less than cur value.
    pub w: usize,
    pub m: usize,
    pub gamma: f32,
    // Minimum size of onset diff to consider for tempo.
    pub min_diff: f32,
    // Maximum size of onset diff to consider for tempo.
    pub max_diff: f32,
    // Maximum threshold to consider two diffs as the same.
    pub threshold: f32,
    // Rate of decay for cluster values.
    pub decay: f32,
}

impl ValidatedConfig for BeatTrackingConfig {
    fn validate(&self) -> Result<(), String> {
        assertion(
            self.alpha >= 0.0 && self.alpha <= 1.0,
            "alpha must be in [0, 1]",
        )?;
        assertion(
            self.decay >= 0.0 && self.decay <= 1.0,
            "decay must be in [0, 1]",
        )?;
        assertion(self.min_diff > 0.0, "min_diff must be greater than 0")?;
        assertion(self.max_diff > 0.0, "max_diff must be greater than 0")?;
        assertion(
            self.min_diff <= self.max_diff,
            "min_diff must be <= max_diff",
        )?;
        assertion(self.threshold > 0.0, "threshold must be greater than 0")?;
        Ok(())
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub spectrogram: SpectrogramConfig,
    pub beat_tracking: BeatTrackingConfig,
}

impl ValidatedConfig for Config {
    fn validate(&self) -> Result<(), String> {
        match self.spectrogram.validate() {
            Ok(()) => Ok(()),
            Err(e) => Err(format!("error validation spectrogram config: {}", e)),
        }?;
        match self.beat_tracking.validate() {
            Ok(()) => Ok(()),
            Err(e) => Err(format!("error validation spectrogram config: {}", e)),
        }?;
        Ok(())
    }
}

pub fn from_yaml_reader<T, R>(r: R) -> Result<T, String>
where
    T: ValidatedConfig + DeserializeOwned,
    R: Read,
{
    let config: T = match serde_yaml::from_reader(r) {
        Ok(config) => Ok(config),
        Err(e) => Err(format!("error parsing config: {}", e)),
    }?;
    match config.validate() {
        Ok(()) => Ok(()),
        Err(e) => Err(format!("error validating spectrogram config: {}", e)),
    }?;
    Ok(config)
}
