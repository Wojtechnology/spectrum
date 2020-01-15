extern crate gfx_hal as hal;

use std::cell::RefCell;
use std::rc::Rc;

use hal::format as f;
use hal::image as i;
use hal::pass;
use hal::prelude::*;
use hal::Backend;

use crate::device_state::DeviceState;

pub struct RenderPassState<B: Backend> {
    pub render_pass: Option<B::RenderPass>,
    device: Rc<RefCell<DeviceState<B>>>,
}

impl<B: Backend> RenderPassState<B> {
    pub unsafe fn new(format: f::Format, device: Rc<RefCell<DeviceState<B>>>) -> Self {
        let render_pass = {
            let attachment = pass::Attachment {
                format: Some(format),
                samples: 1,
                ops: pass::AttachmentOps::new(
                    pass::AttachmentLoadOp::Clear,
                    pass::AttachmentStoreOp::Store,
                ),
                stencil_ops: pass::AttachmentOps::DONT_CARE,
                layouts: i::Layout::Undefined..i::Layout::Present,
            };

            let subpass = pass::SubpassDesc {
                colors: &[(0, i::Layout::ColorAttachmentOptimal)],
                depth_stencil: None,
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            device
                .borrow()
                .device
                .create_render_pass(&[attachment], &[subpass], &[])
                .ok()
        };

        RenderPassState {
            render_pass,
            device,
        }
    }
}

impl<B: Backend> Drop for RenderPassState<B> {
    fn drop(&mut self) {
        let device = &self.device.borrow().device;
        unsafe {
            device.destroy_render_pass(self.render_pass.take().unwrap());
        }
    }
}
