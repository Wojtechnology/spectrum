use std::fs::File;
use std::io::BufReader;

use obj::{load_obj, Obj};

use crate::gx_object::{TriIndex, TriIndices, Vertex, Vertices};

pub struct GxModel {
    pub vertices: Vertices<f32>,
    pub indices: TriIndices<u16>,
}

fn grouped_map<I, O, F: Fn(Vec<I>) -> O>(v: Vec<I>, k: usize, f: F) -> Vec<O> {
    assert!(
        v.len() % k == 0,
        format!("v's length must be divisible by k: {} vs {}", v.len(), k)
    );
    let mut out = Vec::new();
    let mut intermediate = Vec::new();
    let mut idx = 0;
    for elem in v.into_iter() {
        intermediate.push(elem);
        idx += 1;
        if idx % k == 0 {
            out.push(f(intermediate));
            intermediate = Vec::new()
        }
    }
    out
}

impl GxModel {
    pub fn from_obj(path: String) -> Result<GxModel, String> {
        let file = match File::open(&path) {
            Ok(f) => Ok(f),
            Err(e) => Err(format!("Couldn't open file {}: {:?}", path, e)),
        }?;
        let input = BufReader::new(file);
        let model: Obj = match load_obj(input) {
            Ok(obj) => Ok(obj),
            Err(e) => Err(format!("{} is an invalid obj file: {:?}", path, e)),
        }?;
        let indices = TriIndices::from_vec(grouped_map(model.indices, 3, |idxs| {
            TriIndex::new(idxs[0], idxs[1], idxs[2])
        }));
        let vertices = Vertices::from_vec(
            model
                .vertices
                .iter()
                .map(|vert| {
                    Vertex::new(
                        vert.position[0],
                        vert.position[1],
                        vert.position[2],
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        1.0,
                    )
                })
                .collect(),
        );
        Ok(GxModel { indices, vertices })
    }
}
