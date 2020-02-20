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

pub fn cube_vertices() -> Box<[VertexData<f32>]> {
    let mut vertices = Vertices::<f32>::new();
    // Face 1 (front)
    vertices.add(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0); /* bottom left */
    vertices.add(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0); /* top left */
    vertices.add(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0); /* bottom right */
    vertices.add(1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0); /* top right */
    // Face 2 (top)
    vertices.add(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0); /* bottom left */
    vertices.add(0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0); /* top left */
    vertices.add(1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0); /* bottom right */
    vertices.add(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0); /* top right */
    // Face 3 (back)
    vertices.add(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0); /* bottom left */
    vertices.add(0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0); /* top left */
    vertices.add(1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0); /* bottom right */
    vertices.add(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0); /* top right */
    // Face 4 (bottom)
    vertices.add(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0); /* bottom left */
    vertices.add(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0); /* top left */
    vertices.add(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0); /* bottom right */
    vertices.add(1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0); /* top right */
    // Face 5 (left)
    vertices.add(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0); /* bottom left */
    vertices.add(0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0); /* top left */
    vertices.add(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0); /* bottom right */
    vertices.add(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0); /* top right */
    // Face 6 (right)
    vertices.add(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0); /* bottom left */
    vertices.add(1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0); /* top left */
    vertices.add(1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0); /* bottom right */
    vertices.add(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0); /* top right */
    vertices.data()
}

pub fn cube_indices() -> Box<[TriIndexData<u16>]> {
    let mut indices = TriIndices::<u16>::new();
    // front
    indices.add(0, 1, 2);
    indices.add(2, 1, 3);
    // top
    indices.add(4, 5, 6);
    indices.add(7, 6, 5);
    // back
    indices.add(10, 9, 8);
    indices.add(9, 10, 11);
    // bottom
    indices.add(12, 14, 13);
    indices.add(15, 13, 14);
    // left
    indices.add(16, 17, 18);
    indices.add(19, 18, 17);
    // right
    indices.add(20, 21, 22);
    indices.add(23, 22, 21);
    indices.data()
}
