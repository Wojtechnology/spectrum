#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub a_pos: [f32; 2],
    pub a_uv: [f32; 2],
}

#[derive(Debug, Clone, Copy)]
pub struct Quad {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Quad {
    pub fn vertex_attributes(self) -> [f32; 4 * (2 + 3 + 2)] {
        let x = self.x;
        let y = self.y;
        let w = self.w;
        let h = self.h;
        #[cfg_attr(rustfmt, rustfmt_skip)]
        [
        // X    Y    R    G    B                  U    V
        x  , y+h, 1.0, 0.0, 0.0, /* red     */ 0.0, 1.0, /* bottom left */
        x  , y  , 0.0, 1.0, 0.0, /* green   */ 0.0, 0.0, /* top left */
        x+w, y  , 0.0, 0.0, 1.0, /* blue    */ 1.0, 0.0, /* top right */
        x+w, y+h, 1.0, 0.0, 1.0, /* magenta */ 1.0, 1.0, /* bottom right */
        ]
    }
}
