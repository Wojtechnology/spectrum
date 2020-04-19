use std::time::SystemTime;

pub struct CyclicalBuffer<T: Copy> {
    buf: Vec<T>,
    size: usize,
    cursor: usize,
}

impl<T: Copy> CyclicalBuffer<T> {
    pub fn new(size: usize) -> Self {
        assert!(size > 0, "Size must be greater than 0");
        Self {
            buf: Vec::with_capacity(size),
            size,
            cursor: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn push(&mut self, val: T) {
        if self.len() < self.size {
            self.buf.push(val);
        } else {
            self.buf[self.cursor] = val;
            self.cursor = (self.cursor + 1) % self.size;
        }
        assert!(self.len() <= self.size, "Length must be less than size");
    }

    pub fn get_values(&self) -> Vec<T> {
        assert!(
            self.len() == self.size,
            "Can only get values when buffer is full"
        );
        [&self.buf[self.cursor..], &self.buf[..self.cursor]].concat()
    }
}
