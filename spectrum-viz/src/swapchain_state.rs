extern crate gfx_hal as hal;

use std::cell::RefCell;
use std::rc::Rc;

use hal::format as f;
use hal::image as i;
use hal::prelude::*;
use hal::window as w;
use hal::Backend;

use crate::backend_state::BackendState;
use crate::device_state::DeviceState;
use crate::gx_constant::DIMS;

pub struct SwapchainState<B: Backend> {
    pub swapchain: Option<B::Swapchain>,
    pub backbuffer: Option<Vec<B::Image>>,
    device: Rc<RefCell<DeviceState<B>>>,
    pub extent: i::Extent,
    pub format: f::Format,
}

impl<B: Backend> SwapchainState<B> {
    pub unsafe fn new(backend: &mut BackendState<B>, device: Rc<RefCell<DeviceState<B>>>) -> Self {
        let caps = backend
            .surface
            .capabilities(&device.borrow().physical_device);
        let formats = backend
            .surface
            .supported_formats(&device.borrow().physical_device);
        println!("formats: {:?}", formats);
        let format = formats.map_or(f::Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == f::ChannelType::Srgb)
                .map(|format| *format)
                .unwrap_or(formats[0])
        });

        println!("Surface format: {:?}", format);
        let swap_config = w::SwapchainConfig::from_caps(&caps, format, DIMS);
        let extent = swap_config.extent.to_extent();
        let (swapchain, backbuffer) = device
            .borrow()
            .device
            .create_swapchain(&mut backend.surface, swap_config, None)
            .expect("Can't create swapchain");

        let swapchain = SwapchainState {
            swapchain: Some(swapchain),
            backbuffer: Some(backbuffer),
            device,
            extent,
            format,
        };
        swapchain
    }
}

impl<B: Backend> Drop for SwapchainState<B> {
    fn drop(&mut self) {
        unsafe {
            self.device
                .borrow()
                .device
                .destroy_swapchain(self.swapchain.take().unwrap());
        }
    }
}
