use std::fs::{read_dir, remove_file};

use protobuf_codegen_pure;

fn clean_proto() {
    let to_remove = match read_dir("src/model") {
        Ok(files) => files,
        Err(e) => panic!("Error while reading src/model: {:?}", e),
    };
    for file_res in to_remove {
        match file_res {
            Ok(file) => match remove_file(file.path()) {
                Ok(()) => {}
                Err(e) => panic!("Error removing {:?}: {:?}", file.path(), e),
            },
            Err(e) => panic!("Error while reading src/model: {:?}", e),
        }
    }
}

fn build_proto() {
    protobuf_codegen_pure::Codegen::new()
        .out_dir("src/model")
        .includes(&["proto"])
        .input("proto/data.proto")
        .run()
        .unwrap();
}

fn main() {
    clean_proto();
    build_proto();
}
