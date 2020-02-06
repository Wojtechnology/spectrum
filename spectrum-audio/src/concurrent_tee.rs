use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Debug)]
struct ConcurrentTeeBuffer<A, I> {
    backlog: VecDeque<A>,
    iter: I,
    owner_id: bool,
}

#[derive(Debug)]
pub struct ConcurrentTee<I>
where
    I: Iterator,
{
    arcbuffer: Arc<Mutex<ConcurrentTeeBuffer<I::Item, I>>>,
    id: bool,
}

impl<I> ConcurrentTee<I>
where
    I: Iterator,
{
    pub fn new(iter: I) -> (ConcurrentTee<I>, ConcurrentTee<I>) {
        let buffer = ConcurrentTeeBuffer {
            backlog: VecDeque::new(),
            iter,
            owner_id: false,
        };
        let tee_one = ConcurrentTee {
            arcbuffer: Arc::new(Mutex::new(buffer)),
            id: false,
        };
        let tee_two = ConcurrentTee {
            arcbuffer: tee_one.arcbuffer.clone(),
            id: true,
        };
        (tee_one, tee_two)
    }
}

impl<I> Iterator for ConcurrentTee<I>
where
    I: Iterator,
    I::Item: Copy,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buffer = self.arcbuffer.lock().unwrap();
        if buffer.owner_id == self.id {
            match buffer.backlog.pop_front() {
                None => {}
                some_elt => return some_elt,
            }
        }
        match buffer.iter.next() {
            None => None,
            Some(elt) => {
                buffer.backlog.push_back(elt);
                buffer.owner_id = !self.id;
                Some(elt)
            }
        }
    }
}
