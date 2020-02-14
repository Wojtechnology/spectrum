use gfx_hal as hal;
use hal::format as f;
use hal::image as i;

use crate::gx_object::{TriIndexData, TriIndices, VertexData, Vertices};

pub const COLOR_RANGE: i::SubresourceRange = i::SubresourceRange {
    aspects: f::Aspects::COLOR,
    levels: 0..1,
    layers: 0..1,
};

pub fn triangle_vertices() -> Box<[VertexData<f32>]> {
    let mut vertices = Vertices::<f32>::new();
    vertices.add(-1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
    vertices.add(-1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
    vertices.add(1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    vertices.add(1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0);
    vertices.data()
}

pub fn triangle_indices() -> Box<[TriIndexData<u16>]> {
    let mut indices = TriIndices::<u16>::new();
    indices.add(0, 1, 2);
    indices.add(0, 1, 3);
    indices.add(0, 2, 3);
    indices.add(1, 2, 3);
    indices.data()
}
