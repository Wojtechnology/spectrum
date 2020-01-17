extern crate gfx_hal as hal;

use hal::format as f;
use hal::image as i;

use crate::gx_object::{Quad, Vertex};

pub const COLOR_RANGE: i::SubresourceRange = i::SubresourceRange {
    aspects: f::Aspects::COLOR,
    levels: 0..1,
    layers: 0..1,
};

pub const TRIANGLE: [[f32; 5]; 3] = [
    [-0.5, 0.5, 1.0, 0.0, 0.0],
    [-0.5, -0.5, 0.0, 1.0, 0.0],
    [0.5, -0.33, 0.0, 0.0, 1.0],
];

pub const QUAD: [Vertex; 6] = [
    Vertex {
        a_pos: [-0.5, 0.33],
        a_uv: [0.0, 1.0],
    },
    Vertex {
        a_pos: [0.5, 0.33],
        a_uv: [1.0, 1.0],
    },
    Vertex {
        a_pos: [0.5, -0.33],
        a_uv: [1.0, 0.0],
    },
    Vertex {
        a_pos: [-0.5, 0.33],
        a_uv: [0.0, 1.0],
    },
    Vertex {
        a_pos: [0.5, -0.33],
        a_uv: [1.0, 0.0],
    },
    Vertex {
        a_pos: [-0.5, -0.33],
        a_uv: [0.0, 0.0],
    },
];

pub const ACTUAL_QUAD: Quad = Quad {
    x: 0.0,
    y: 0.0,
    w: 1.0,
    h: 1.0,
};
