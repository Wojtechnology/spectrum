extern crate gfx_hal as hal;

use hal::adapter::Adapter;
use hal::prelude::*;
use hal::queue::QueueGroup;
use hal::Backend;

pub struct DeviceState<B: Backend> {
    pub device: B::Device,
    pub physical_device: B::PhysicalDevice,
    pub queues: QueueGroup<B>,
}

impl<B: Backend> DeviceState<B> {
    pub fn new(adapter: Adapter<B>, surface: &B::Surface) -> Self {
        let family = adapter
            .queue_families
            .iter()
            .find(|family| {
                surface.supports_queue_family(family) && family.queue_type().supports_graphics()
            })
            .unwrap();
        let mut gpu = unsafe {
            adapter
                .physical_device
                .open(&[(family, &[1.0])], hal::Features::empty())
                .unwrap()
        };

        DeviceState {
            device: gpu.device,
            queues: gpu.queue_groups.pop().unwrap(),
            physical_device: adapter.physical_device,
        }
    }
}
