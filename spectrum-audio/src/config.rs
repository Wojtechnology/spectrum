use serde::{Deserialize, Serialize};
use serde_yaml;

#[derive(Debug, Serialize, Deserialize)]
pub struct SubsetConfig {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SpectrogramConfig {
    // y-intercept of the sine filter applied to the signal before it is passed into the FFT.
    pub sine_filter_y_int: f32,
    // Size of circular buffer for audio data points and as a result, number of data points that
    // are input into the FFT transform. The output of the transform will be of the same size but
    // the number of usable output datapoints are only the values for positive coefficients, which
    // means that we can only use `buffer_size / 2`.
    pub buffer_size: usize,
    // Every `stutter_size` data points, we run the FFT transform on the current circular buffer
    // (assuming there are currently enough values in it).
    pub stutter_size: usize,
    // Size of window for averaging frequency bins. If this is set to 1, will not do any averaging,
    // which is currently preferred.
    pub window_size: usize,
    // What subset of bands to include in the output. Usually you want this to be buffer_size / 2
    // so that you pick up only the positive coefficients. Note that this is applied AFTER
    // windowing.
    pub band_subset: Option<SubsetConfig>,
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
        assertion(
            self.sine_filter_y_int >= 0.0,
            "sine_filter_y_int must be greater or equal to 0",
        )?;
        assertion(
            self.sine_filter_y_int <= 1.0,
            "sine_filter_y_int must be less or equal to 1",
        )?;
        assertion(self.buffer_size > 0, "buffer_size must be greater than 0")?;
        assertion(self.buffer_size > 0, "buffer_size must be greater than 0")?;
        assertion(self.stutter_size > 0, "stutter_size must be greater than 0")?;
        assertion(self.window_size > 0, "window_size must be greater than 0")?;
        assertion(
            self.buffer_size % 2 == 0,
            "buffer_size must be divisible by 2",
        )?;
        assertion(
            self.buffer_size % self.window_size == 0,
            "buffer_size must be divisible by window_size",
        )?;
        match &self.band_subset {
            Some(subset) => {
                assertion(
                    subset.start <= subset.end,
                    "band_subset.start must be at most band_subset.end",
                )?;
                assertion(
                    subset.end <= self.buffer_size / self.window_size,
                    "band_subset.end must be at most buffer_size",
                )?;
                Result::<(), String>::Ok(())
            }
            None => Ok(()),
        }?;
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
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
