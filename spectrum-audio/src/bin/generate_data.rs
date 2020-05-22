use std::env;
use std::fs::File;
use std::process;

use spectrum_serdes::sparse::SparseStreamWriter;

use spectrum_audio::audio_loop::generate_data;
use spectrum_audio::config::{from_yaml_reader, SpectrogramConfig};
use spectrum_audio::mp3::Mp3Decoder;
use spectrum_audio::raw_stream::RawStream;

// TODO: Should really replace all of this with some command line arg parsing library.
fn eprint_usage_and_exit() -> ! {
    let mut args = env::args();
    let program_name = args
        .next()
        .expect("program name must exist, otherwise rust is broken");
    eprintln!(
        "usage: {} <config_path> <audio_path> <write_path>",
        program_name
    );
    process::exit(1);
}

fn get_arg(args: &mut env::Args, err_msg: &'static str) -> Result<String, &'static str> {
    match args.next() {
        Some(arg) => Ok(arg),
        None => Err(err_msg),
    }
}

fn open_file(path: &str) -> Result<File, String> {
    match File::open(&path) {
        Ok(reader) => Ok(reader),
        Err(e) => Err(format!("could not open {}: {}", path, e)),
    }
}

struct Args {
    config_path: String,
    audio_path: String,
    write_path: String,
}

impl Args {
    fn from_env_args() -> Result<Self, String> {
        let mut args = env::args();
        args.next(); // program name

        let config_path = get_arg(&mut args, "missing config_path")?;
        let audio_path = get_arg(&mut args, "missing audio_path")?;
        let write_path = get_arg(&mut args, "missing write_path")?;

        Ok(Self {
            config_path,
            audio_path,
            write_path,
        })
    }
}

fn init() -> Result<(Mp3Decoder<File>, SpectrogramConfig, File), String> {
    let args = Args::from_env_args()?;

    let audio_reader = open_file(&args.audio_path)?;
    let decoder = match Mp3Decoder::new(audio_reader) {
        Ok(decoder) => Ok(decoder),
        Err(e) => Err(format!("{}", e)),
    }?;

    let config_reader = open_file(&args.config_path)?;
    let config = from_yaml_reader::<SpectrogramConfig, File>(config_reader)?;
    println!("{:?}", config);

    let writer = match File::create(&args.write_path) {
        Ok(file) => Ok(file),
        Err(e) => Err(format!("Error creating write file: {:?}", e)),
    }?;

    Ok((decoder, config, writer))
}

fn main() {
    let (decoder, config, mut writer) = match init() {
        Ok((decoder, config, writer)) => (decoder, config, writer),
        Err(e) => {
            eprintln!("{}", e);
            eprint_usage_and_exit();
        }
    };
    let channels = match config.band_subset {
        Some(band_subset) => band_subset.end - band_subset.start,
        None => config.buffer_size,
    } as u32;
    let sample_rate = decoder.sample_rate() as f64;
    let mut sparse_writer = SparseStreamWriter::new(channels, sample_rate);

    let data = generate_data(decoder, &config);

    let mut idx = config.buffer_size as u64;
    let incr = config.stutter_size as u64;
    for row in data.iter() {
        sparse_writer.push(idx, row.clone());
        idx += incr;
    }

    match sparse_writer.write(&mut writer) {
        Ok(()) => (),
        Err(e) => panic!("Failed writing spectrogram file: {:?}", e),
    }
}
