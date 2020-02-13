use gfx_hal as hal;
use hal::format as f;
use hal::image as i;

use crate::gx_object::{VertexData, Vertices};

pub const COLOR_RANGE: i::SubresourceRange = i::SubresourceRange {
    aspects: f::Aspects::COLOR,
    levels: 0..1,
    layers: 0..1,
};

pub fn triangle() -> Box<[VertexData<f32>]> {
    let mut vertices = Vertices::<f32>::new();
    vertices.add(-0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
    vertices.add(-0.5, -0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
    vertices.add(0.5, -0.33, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    vertices.data()
}
