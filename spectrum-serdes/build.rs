use std::fs::{read_dir, remove_file, write};

use protobuf_codegen_pure;

const PROTOS: &[&str] = &["data", "sparse_stream"];
const PROTO_DIR: &str = "proto";
const MODEL_DIR: &str = "src/model";

fn clean_proto() {
    let to_remove = match read_dir(MODEL_DIR) {
        Ok(files) => files,
        Err(e) => panic!("Error while reading {}: {:?}", MODEL_DIR, e),
    };
    for file_res in to_remove {
        match file_res {
            Ok(file) => match remove_file(file.path()) {
                Ok(()) => {}
                Err(e) => panic!("Error removing {:?}: {:?}", file.path(), e),
            },
            Err(e) => panic!("Error while reading {}: {:?}", MODEL_DIR, e),
        }
    }
}

fn build_proto() {
    let proto_files: Vec<_> = PROTOS
        .iter()
        .map(|name| format!("{}/{}.proto", PROTO_DIR, name))
        .collect();

    protobuf_codegen_pure::Codegen::new()
        .out_dir(MODEL_DIR)
        .includes(&[PROTO_DIR])
        .inputs(proto_files)
        .run()
        .unwrap();
}

fn build_mod() {
    // Write the mod.rs file required for loading protos into lib
    let mut mod_text = String::new();
    for proto in PROTOS {
        mod_text.push_str(&format!("pub mod {};\n", proto));
    }
    let mod_path = format!("{}/mod.rs", MODEL_DIR);
    match write(&mod_path, mod_text) {
        Ok(()) => {}
        Err(e) => panic!("Error while writing {}: {:?}", mod_path, e),
    }
}

fn main() {
    clean_proto();
    build_proto();
    build_mod();
}
