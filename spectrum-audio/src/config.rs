use serde::{Deserialize, Serialize};
use serde_yaml;

#[derive(Debug, Serialize, Deserialize)]
pub struct SpectrogramConfig {
    // Size of circular buffer for audio data points and as a result, number of data points that
    // are input into the FFT transform. The output of the transform will be of the same size but
    // the number of usable output datapoints are only the values for positive coefficients, which
    // means that we can only use `buffer_size / 2`.
    buffer_size: usize,
    // Every `stutter_size` data points, we run the FFT transform on the current circular buffer
    // (assuming there are currently enough values in it).
    stutter_size: usize,
    // Size of window for averaging frequency bins. If this is set to 1, will not do any averaging,
    // which is currently preferred.
    window_size: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    spectrogram: SpectrogramConfig,
}

impl Config {
    pub fn from_yaml(s: &str) -> Result<Self, String> {
        match serde_yaml::from_str(s) {
            Ok(config) => Ok(config),
            Err(e) => Err(format!("error parsing config: {}", e)),
        }
    }
}
