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

pub type VertexData<T> = [T; 8];
pub type TriIndexData<T> = [T; 3];

#[derive(Debug, Clone, Copy)]
pub struct Vertex<T>
where
    T: Copy,
{
    pos: Position3D<T>,
    col: ColorRGB<T>,
    uv: UVMapping<T>,
}

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

#[derive(Debug, Clone)]
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

    pub fn from_vec(vertices: Vec<Vertex<T>>) -> Self {
        Vertices { vertices }
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
        self.vertices
            .iter()
            .map(|&vertex| vertex.data())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TriIndex<T>
where
    T: Copy,
{
    pt1: T,
    pt2: T,
    pt3: T,
}

impl<T> TriIndex<T>
where
    T: Copy,
{
    pub fn new(pt1: T, pt2: T, pt3: T) -> Self {
        TriIndex { pt1, pt2, pt3 }
    }

    pub fn data(&self) -> TriIndexData<T> {
        [self.pt1, self.pt2, self.pt3]
    }
}

#[derive(Debug, Clone)]
pub struct TriIndices<T>
where
    T: Copy,
{
    indices: Vec<TriIndex<T>>,
}

impl<T> TriIndices<T>
where
    T: Copy,
{
    pub fn new() -> Self {
        TriIndices {
            indices: Vec::<TriIndex<T>>::new(),
        }
    }

    pub fn from_vec(indices: Vec<TriIndex<T>>) -> Self {
        TriIndices { indices }
    }

    pub fn add(&mut self, pt1: T, pt2: T, pt3: T) {
        self.indices.push(TriIndex::new(pt1, pt2, pt3));
    }

    pub fn data(&self) -> Box<[TriIndexData<T>]> {
        self.indices
            .iter()
            .map(|&index| index.data())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }
}
