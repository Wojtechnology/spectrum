extern crate gfx_hal as hal;

use std::cell::RefCell;
use std::rc::Rc;

use hal::adapter::MemoryType;
use hal::buffer;
use hal::pso;
use hal::Backend;

use crate::buffer_state::BufferState;
use crate::desc_set::{DescSet, DescSetWrite};
use crate::device_state::DeviceState;

pub struct Uniform<B: Backend> {
    pub buffer: Option<BufferState<B>>,
    pub desc: Option<DescSet<B>>,
}

impl<B: Backend> Uniform<B> {
    pub unsafe fn new<T>(
        device: Rc<RefCell<DeviceState<B>>>,
        memory_types: &[MemoryType],
        data: &[T],
        mut desc: DescSet<B>,
        binding: u32,
    ) -> Self
    where
        T: Copy,
    {
        let buffer = BufferState::new(
            Rc::clone(&device),
            &data,
            buffer::Usage::UNIFORM,
            memory_types,
        );
        let buffer = Some(buffer);

        desc.write_to_state(
            vec![DescSetWrite {
                binding,
                array_offset: 0,
                descriptors: Some(pso::Descriptor::Buffer(
                    buffer.as_ref().unwrap().get_buffer(),
                    None..None,
                )),
            }],
            &mut device.borrow_mut().device,
        );

        Uniform {
            buffer,
            desc: Some(desc),
        }
    }

    pub fn get_layout(&self) -> &B::DescriptorSetLayout {
        self.desc.as_ref().unwrap().get_layout()
    }
}
