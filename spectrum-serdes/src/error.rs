#[derive(Debug)]
pub struct WriteError {
    pub msg: String,
}

impl WriteError {
    pub fn new(msg: &str) -> Self {
        Self {
            msg: String::from(msg),
        }
    }
}

#[derive(Debug)]
pub struct ReadError {
    pub msg: String,
}

impl ReadError {
    pub fn new(msg: &str) -> Self {
        Self {
            msg: String::from(msg),
        }
    }
}
