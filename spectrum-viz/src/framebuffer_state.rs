extern crate gfx_hal as hal;

use std::cell::RefCell;
use std::iter;
use std::rc::Rc;

use hal::command;
use hal::pool;
use hal::prelude::*;
use hal::Backend;

use crate::device_state::DeviceState;

const MAX_FRAMES_IN_FLIGHT: usize = 3;

pub struct FramebufferState<B: Backend> {
    command_pools: Vec<B::CommandPool>,
    command_buffers: Vec<B::CommandBuffer>,
    present_fences: Vec<B::Fence>,
    present_semaphores: Vec<B::Semaphore>,
    device: Rc<RefCell<DeviceState<B>>>,
    frames_in_flight: usize,
    frame_num: usize,
}

impl<B: Backend> FramebufferState<B> {
    pub unsafe fn new(device: Rc<RefCell<DeviceState<B>>>) -> Self {
        // TODO: Figure out if the actual number can be computed.
        let frames_in_flight = MAX_FRAMES_IN_FLIGHT;

        let mut command_pools = Vec::with_capacity(frames_in_flight);
        let mut command_buffers = Vec::with_capacity(frames_in_flight);
        let mut present_fences = Vec::with_capacity(frames_in_flight);
        let mut present_semaphores = Vec::with_capacity(frames_in_flight);

        for _ in 0..frames_in_flight {
            let mut command_pool = device
                .borrow()
                .device
                .create_command_pool(
                    device.borrow().queues.family,
                    pool::CommandPoolCreateFlags::empty(),
                )
                .expect("Can't create command pool");
            command_buffers.push(command_pool.allocate_one(command::Level::Primary));
            command_pools.push(command_pool);
            present_fences.push(device.borrow().device.create_fence(true).unwrap());
            present_semaphores.push(device.borrow().device.create_semaphore().unwrap());
        }

        FramebufferState {
            command_pools,
            command_buffers,
            present_fences,
            present_semaphores,
            device,
            frames_in_flight,
            frame_num: 0,
        }
    }

    pub fn next_frame_info(
        &mut self,
    ) -> (
        &mut B::CommandPool,
        &mut B::CommandBuffer,
        &B::Fence,
        &B::Semaphore,
    ) {
        let cur_frame_idx = self.frame_num % self.frames_in_flight;
        self.frame_num += 1;

        (
            &mut self.command_pools[cur_frame_idx],
            &mut self.command_buffers[cur_frame_idx],
            &self.present_fences[cur_frame_idx],
            &self.present_semaphores[cur_frame_idx],
        )
    }
}

impl<B: Backend> Drop for FramebufferState<B> {
    fn drop(&mut self) {
        let device = &self.device.borrow().device;

        unsafe {
            for fence in self.present_fences.drain(..) {
                device.wait_for_fence(&fence, !0).unwrap();
                device.destroy_fence(fence);
            }

            for (mut command_pool, command_buffer) in self
                .command_pools
                .drain(..)
                .zip(self.command_buffers.drain(..))
            {
                command_pool.free(iter::once(command_buffer));
                device.destroy_command_pool(command_pool);
            }

            for present_semaphore in self.present_semaphores.drain(..) {
                device.destroy_semaphore(present_semaphore);
            }
        }
    }
}
