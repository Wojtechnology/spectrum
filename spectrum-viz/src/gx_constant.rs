extern crate gfx_hal as hal;

use hal::window as w;

use crate::gx_object::{Quad, Vertex};

// TODO: Move into another constants file
pub const DIMS: w::Extent2D = w::Extent2D {
    width: 1024,
    height: 768,
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
