#[derive(Debug, Clone, Copy)]
struct Position3D<T> {
    x: T,
    y: T,
    z: T,
}

#[derive(Debug, Clone, Copy)]
struct ColorRGB<T> {
    r: T,
    g: T,
    b: T,
}

#[derive(Debug, Clone, Copy)]
struct UVMapping<T> {
    u: T,
    v: T,
}

#[derive(Debug, Clone, Copy)]
struct Vertex<T>
where
    T: Copy,
{
    pos: Position3D<T>,
    col: ColorRGB<T>,
    uv: UVMapping<T>,
}

pub type VertexData<T> = [T; 8];

impl<T> Vertex<T>
where
    T: Copy,
{
    pub fn new(
        pos_x: T,
        pos_y: T,
        pos_z: T,
        col_r: T,
        col_g: T,
        col_b: T,
        uv_u: T,
        uv_v: T,
    ) -> Self {
        Vertex {
            pos: Position3D {
                x: pos_x,
                y: pos_y,
                z: pos_z,
            },
            col: ColorRGB {
                r: col_r,
                g: col_g,
                b: col_b,
            },
            uv: UVMapping { u: uv_u, v: uv_v },
        }
    }

    pub fn data(&self) -> VertexData<T> {
        [
            self.pos.x, self.pos.y, self.pos.z, self.col.r, self.col.g, self.col.b, self.uv.u,
            self.uv.v,
        ]
    }
}

pub struct Vertices<T>
where
    T: Copy,
{
    vertices: Vec<Vertex<T>>,
}

impl<T> Vertices<T>
where
    T: Copy,
{
    pub fn new() -> Self {
        Vertices {
            vertices: Vec::<Vertex<T>>::new(),
        }
    }

    pub fn add(
        &mut self,
        pos_x: T,
        pos_y: T,
        pos_z: T,
        col_r: T,
        col_g: T,
        col_b: T,
        uv_u: T,
        uv_v: T,
    ) {
        self.vertices.push(Vertex::new(
            pos_x, pos_y, pos_z, col_r, col_g, col_b, uv_u, uv_v,
        ));
    }

    pub fn data(&self) -> Box<[VertexData<T>]> {
        let vertex_datas: Vec<_> = self.vertices.iter().map(|&vertex| vertex.data()).collect();
        vertex_datas.into_boxed_slice()
    }
}
