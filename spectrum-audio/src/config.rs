use serde::{Deserialize, Serialize};
use serde_yaml;

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

impl SpectrogramConfig {
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
pub struct Config {
    pub spectrogram: SpectrogramConfig,
}

impl Config {
    pub fn from_yaml(s: &str) -> Result<Self, String> {
        let config: Config = match serde_yaml::from_str(s) {
            Ok(config) => Ok(config),
            Err(e) => Err(format!("error parsing config: {}", e)),
        }?;
        match config.spectrogram.validate() {
            Ok(()) => Ok(()),
            Err(e) => Err(format!("error validation spectrogram config: {}", e)),
        }?;
        Ok(config)
    }
}
