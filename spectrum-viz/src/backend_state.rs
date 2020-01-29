use std::mem::ManuallyDrop;
use std::ptr;

#[cfg(not(any(feature = "vulkan", feature = "metal",)))]
use gfx_backend_empty as back;
#[cfg(feature = "metal")]
use gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as back;
use gfx_hal as hal;
use hal::prelude::*;
use hal::Backend;

use crate::adapter_state::AdapterState;

pub struct BackendState<B: Backend> {
    instance: Option<B::Instance>,
    pub surface: ManuallyDrop<B::Surface>,
    pub adapter: AdapterState<B>,
    pub window: winit::window::Window,
}

#[cfg(any(feature = "vulkan", feature = "metal"))]
pub fn create_backend(
    wb: winit::window::WindowBuilder,
    event_loop: &winit::event_loop::EventLoop<()>,
) -> BackendState<back::Backend> {
    let window = wb.build(event_loop).unwrap();
    let instance =
        back::Instance::create("gfx-rs colour-uniform", 1).expect("Failed to create an instance!");
    let surface = unsafe {
        instance
            .create_surface(&window)
            .expect("Failed to create a surface!")
    };
    let mut adapters = instance.enumerate_adapters();
    BackendState {
        instance: Some(instance),
        adapter: AdapterState::new(&mut adapters),
        surface: ManuallyDrop::new(surface),
        window,
    }
}

impl<B: Backend> Drop for BackendState<B> {
    fn drop(&mut self) {
        if let Some(instance) = &self.instance {
            unsafe {
                let surface = ManuallyDrop::into_inner(ptr::read(&self.surface));
                instance.destroy_surface(surface);
            }
        }
    }
}
