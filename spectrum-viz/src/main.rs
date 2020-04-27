use std::env;
use std::fs::File;
use std::io::Read;
use std::process;
use std::sync::{Arc, RwLock};
use std::thread;

use spectrum_audio::audio_loop::run_audio_loop;
use spectrum_audio::config::Config;
use spectrum_audio::mp3::Mp3Decoder;
use spectrum_audio::shared_data::SharedData;
use spectrum_viz::event_loop;

// TODO: Should really replace all of this with some command line arg parsing library.
fn eprint_usage_and_exit() -> ! {
    let mut args = env::args();
    let program_name = args
        .next()
        .expect("program name must exist, otherwise rust is broken");
    eprintln!("usage: {} <config_path> <audio_path>", program_name);
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
}

impl Args {
    fn from_env_args() -> Result<Self, String> {
        let mut args = env::args();
        args.next(); // program name

        let config_path = get_arg(&mut args, "missing config_path")?;
        let audio_path = get_arg(&mut args, "missing audio_path")?;

        Ok(Self {
            config_path,
            audio_path,
        })
    }
}

fn init() -> Result<(Mp3Decoder<File>, Config), String> {
    let args = Args::from_env_args()?;

    let audio_reader = open_file(&args.audio_path)?;
    let decoder = match Mp3Decoder::new(audio_reader) {
        Ok(decoder) => Ok(decoder),
        Err(e) => Err(format!("{}", e)),
    }?;

    let mut config_reader = open_file(&args.config_path)?;
    let mut config_str = String::new();
    match config_reader.read_to_string(&mut config_str) {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("{}", e)),
    }?;
    let config = Config::from_yaml(&config_str)?;
    println!("{:?}", config);

    Ok((decoder, config))
}

fn main() {
    let (decoder, config) = match init() {
        Ok((decoder, config)) => (decoder, config),
        Err(e) => {
            eprintln!("{}", e);
            eprint_usage_and_exit();
        }
    };

    let shared_data = Arc::new(RwLock::new(SharedData::new(128)));
    let shared_data_clone = shared_data.clone();

    thread::spawn(move || {
        run_audio_loop(decoder, shared_data_clone, config);
    });
    event_loop::run(shared_data);
}
