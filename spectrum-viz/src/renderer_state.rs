extern crate gfx_hal as hal;

use std::cell::RefCell;
use std::io::Cursor;
use std::iter;
use std::rc::Rc;
use std::time::Instant;

use hal::pso::{PipelineStage, ShaderStageFlags};
use hal::{buffer, command, pool, prelude::*, pso, queue::Submission, window as w, Backend};

use crate::backend_state::BackendState;
use crate::buffer_state::BufferState;
use crate::desc_set::DescSetLayout;
use crate::device_state::DeviceState;
use crate::framebuffer_state::FramebufferState;
use crate::gx_constant::{ACTUAL_QUAD, QUAD, TRIANGLE};
use crate::gx_object::Vertex;
use crate::image_state::ImageState;
use crate::pipeline_state::PipelineState;
use crate::render_pass_state::RenderPassState;
use crate::swapchain_state::SwapchainState;
use crate::uniform::Uniform;

pub struct RendererState<B: Backend> {
    uniform_desc_pool: Option<B::DescriptorPool>,
    img_desc_pool: Option<B::DescriptorPool>,
    pub viewport: pso::Viewport,
    creation_instant: Instant,
    // Locally defined data
    pub backend: BackendState<B>,
    render_pass: RenderPassState<B>,
    vertex_buffer: BufferState<B>,
    triangle_vertex_buffer: BufferState<B>,
    quad_vertex_buffer: BufferState<B>,
    image: ImageState<B>,
    pipeline: PipelineState<B>,
    framebuffer: FramebufferState<B>,
    pub uniform: Uniform<B>,
    swapchain: Option<SwapchainState<B>>,
    device: Rc<RefCell<DeviceState<B>>>,
}

impl<B: Backend> RendererState<B> {
    pub unsafe fn new(mut backend: BackendState<B>) -> Self {
        let device = Rc::new(RefCell::new(DeviceState::new(
            backend.adapter.adapter.take().unwrap(),
            &backend.surface,
        )));

        let image_desc = DescSetLayout::new(
            Rc::clone(&device),
            vec![
                pso::DescriptorSetLayoutBinding {
                    binding: 0,
                    ty: pso::DescriptorType::SampledImage,
                    count: 1,
                    stage_flags: ShaderStageFlags::FRAGMENT,
                    immutable_samplers: false,
                },
                pso::DescriptorSetLayoutBinding {
                    binding: 1,
                    ty: pso::DescriptorType::Sampler,
                    count: 1,
                    stage_flags: ShaderStageFlags::FRAGMENT,
                    immutable_samplers: false,
                },
            ],
        );

        let uniform_desc = DescSetLayout::new(
            Rc::clone(&device),
            vec![pso::DescriptorSetLayoutBinding {
                binding: 0,
                ty: pso::DescriptorType::UniformBuffer,
                count: 1,
                stage_flags: ShaderStageFlags::FRAGMENT,
                immutable_samplers: false,
            }],
        );

        let mut img_desc_pool = device
            .borrow()
            .device
            .create_descriptor_pool(
                1, // # of sets
                &[
                    pso::DescriptorRangeDesc {
                        ty: pso::DescriptorType::SampledImage,
                        count: 1,
                    },
                    pso::DescriptorRangeDesc {
                        ty: pso::DescriptorType::Sampler,
                        count: 1,
                    },
                ],
                pso::DescriptorPoolCreateFlags::empty(),
            )
            .ok();

        let mut uniform_desc_pool = device
            .borrow()
            .device
            .create_descriptor_pool(
                1, // # of sets
                &[pso::DescriptorRangeDesc {
                    ty: pso::DescriptorType::UniformBuffer,
                    count: 1,
                }],
                pso::DescriptorPoolCreateFlags::empty(),
            )
            .ok();

        let image_desc = image_desc.create_desc_set(img_desc_pool.as_mut().unwrap());
        let uniform_desc = uniform_desc.create_desc_set(uniform_desc_pool.as_mut().unwrap());

        println!("Memory types: {:?}", backend.adapter.memory_types);

        const IMAGE_LOGO: &'static [u8] = include_bytes!("data/logo.png");
        let img = image::load(Cursor::new(&IMAGE_LOGO[..]), image::PNG)
            .unwrap()
            .to_rgba();

        let mut staging_pool = device
            .borrow()
            .device
            .create_command_pool(
                device.borrow().queues.family,
                pool::CommandPoolCreateFlags::empty(),
            )
            .expect("Can't create staging command pool");

        let image = ImageState::new(
            image_desc,
            &img,
            &backend.adapter,
            buffer::Usage::TRANSFER_SRC,
            &mut device.borrow_mut(),
            &mut staging_pool,
        );

        let quad_vertex_buffer = BufferState::new::<f32>(
            Rc::clone(&device),
            &ACTUAL_QUAD.vertex_attributes(),
            buffer::Usage::VERTEX,
            &backend.adapter.memory_types,
        );

        let triangle_vertex_buffer = BufferState::new::<[f32; 5]>(
            Rc::clone(&device),
            &TRIANGLE,
            buffer::Usage::VERTEX,
            &backend.adapter.memory_types,
        );

        let vertex_buffer = BufferState::new::<Vertex>(
            Rc::clone(&device),
            &QUAD,
            buffer::Usage::VERTEX,
            &backend.adapter.memory_types,
        );

        let uniform = Uniform::new(
            Rc::clone(&device),
            &backend.adapter.memory_types,
            &[1f32, 1.0f32, 1.0f32, 1.0f32],
            uniform_desc,
            0,
        );

        image.wait_for_transfer_completion();

        device.borrow().device.destroy_command_pool(staging_pool);

        let mut swapchain = Some(SwapchainState::new(&mut backend, Rc::clone(&device)));

        let render_pass = RenderPassState::new(swapchain.as_ref().unwrap(), Rc::clone(&device));

        let framebuffer = FramebufferState::new(
            Rc::clone(&device),
            &render_pass,
            swapchain.as_mut().unwrap(),
        );

        let pipeline = PipelineState::new_triangle(
            vec![image.get_layout(), uniform.get_layout()],
            render_pass.render_pass.as_ref().unwrap(),
            Rc::clone(&device),
        );

        let viewport = RendererState::create_viewport(swapchain.as_ref().unwrap());

        RendererState {
            backend,
            device,
            image,
            img_desc_pool,
            uniform_desc_pool,
            vertex_buffer,
            triangle_vertex_buffer,
            quad_vertex_buffer,
            uniform,
            render_pass,
            pipeline,
            swapchain,
            framebuffer,
            viewport,
            creation_instant: std::time::Instant::now(),
        }
    }

    pub fn recreate_swapchain(&mut self) {
        self.device.borrow().device.wait_idle().unwrap();

        self.swapchain.take().unwrap();

        self.swapchain =
            Some(unsafe { SwapchainState::new(&mut self.backend, Rc::clone(&self.device)) });

        self.render_pass = unsafe {
            RenderPassState::new(self.swapchain.as_ref().unwrap(), Rc::clone(&self.device))
        };

        self.framebuffer = unsafe {
            FramebufferState::new(
                Rc::clone(&self.device),
                &self.render_pass,
                self.swapchain.as_mut().unwrap(),
            )
        };

        self.pipeline = unsafe {
            PipelineState::new_triangle(
                vec![self.image.get_layout(), self.uniform.get_layout()],
                self.render_pass.render_pass.as_ref().unwrap(),
                Rc::clone(&self.device),
            )
        };

        self.viewport = Self::create_viewport(self.swapchain.as_ref().unwrap());
    }

    fn create_viewport(swapchain: &SwapchainState<B>) -> pso::Viewport {
        pso::Viewport {
            rect: pso::Rect {
                x: 0,
                y: 0,
                w: swapchain.extent.width as i16,
                h: swapchain.extent.height as i16,
            },
            depth: 0.0..1.0,
        }
    }

    pub fn draw_triangle(&mut self, cr: f32, cg: f32, cb: f32) -> Result<(), &'static str> {
        let sem_index = self.framebuffer.next_acq_pre_pair_index();

        let frame: w::SwapImageIndex = unsafe {
            let (acquire_semaphore, _) = self
                .framebuffer
                .get_frame_data(None, Some(sem_index))
                .1
                .unwrap();
            match self
                .swapchain
                .as_mut()
                .unwrap()
                .swapchain
                .as_mut()
                .unwrap()
                .acquire_image(!0, Some(acquire_semaphore), None)
            {
                Ok((i, _)) => Ok(i),
                Err(_) => Err("Couldn't find frame"),
            }?
        };

        let duration = std::time::Instant::now().duration_since(self.creation_instant);
        let time_f32 = duration.as_secs() as f32 + duration.subsec_nanos() as f32 * 1e-9;

        let (fid, sid) = self
            .framebuffer
            .get_frame_data(Some(frame as usize), Some(sem_index));

        let (framebuffer_fence, framebuffer, command_pool, command_buffers) = fid.unwrap();
        let (image_acquired, image_present) = sid.unwrap();
        unsafe {
            self.device
                .borrow()
                .device
                .wait_for_fence(framebuffer_fence, !0)
                .unwrap();
            self.device
                .borrow()
                .device
                .reset_fence(framebuffer_fence)
                .unwrap();
            command_pool.reset(false);

            // Rendering
            let mut cmd_buffer = match command_buffers.pop() {
                Some(cmd_buffer) => cmd_buffer,
                None => command_pool.allocate_one(command::Level::Primary),
            };
            cmd_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);
            cmd_buffer.set_viewports(0, &[self.viewport.clone()]);
            cmd_buffer.set_scissors(0, &[self.viewport.rect]);
            cmd_buffer.bind_graphics_pipeline(self.pipeline.pipeline.as_ref().unwrap());
            cmd_buffer.bind_vertex_buffers(0, Some((self.triangle_vertex_buffer.get_buffer(), 0)));
            cmd_buffer.push_graphics_constants(
                self.pipeline.pipeline_layout.as_ref().unwrap(),
                ShaderStageFlags::FRAGMENT,
                0,
                &[time_f32.to_bits()],
            );
            cmd_buffer.bind_graphics_descriptor_sets(
                self.pipeline.pipeline_layout.as_ref().unwrap(),
                0,
                vec![
                    self.image.desc.set.as_ref().unwrap(),
                    self.uniform.desc.as_ref().unwrap().set.as_ref().unwrap(),
                ],
                &[],
            ); //TODO

            cmd_buffer.begin_render_pass(
                self.render_pass.render_pass.as_ref().unwrap(),
                &framebuffer,
                self.viewport.rect,
                &[command::ClearValue {
                    color: command::ClearColor {
                        float32: [cr, cg, cb, 1.0],
                    },
                }],
                command::SubpassContents::Inline,
            );
            cmd_buffer.draw(0..3, 0..1);
            cmd_buffer.end_render_pass();
            cmd_buffer.finish();

            let submission = Submission {
                command_buffers: iter::once(&cmd_buffer),
                wait_semaphores: iter::once((&*image_acquired, PipelineStage::BOTTOM_OF_PIPE)),
                signal_semaphores: iter::once(&*image_present),
            };

            self.device.borrow_mut().queues.queues[0].submit(submission, Some(framebuffer_fence));
            command_buffers.push(cmd_buffer);

            // present frame
            match self
                .swapchain
                .as_ref()
                .unwrap()
                .swapchain
                .as_ref()
                .unwrap()
                .present(
                    &mut self.device.borrow_mut().queues.queues[0],
                    frame,
                    Some(&*image_present),
                ) {
                Ok(_) => Ok(()),
                Err(_) => Err("Failed to present"),
            }
        }
    }

    fn draw_rust_logo(&mut self, cr: f32, cg: f32, cb: f32) -> Result<(), &'static str> {
        let sem_index = self.framebuffer.next_acq_pre_pair_index();

        let frame: w::SwapImageIndex = unsafe {
            let (acquire_semaphore, _) = self
                .framebuffer
                .get_frame_data(None, Some(sem_index))
                .1
                .unwrap();
            match self
                .swapchain
                .as_mut()
                .unwrap()
                .swapchain
                .as_mut()
                .unwrap()
                .acquire_image(!0, Some(acquire_semaphore), None)
            {
                Ok((i, _)) => Ok(i),
                Err(_) => Err("Couldn't find frame"),
            }?
        };

        let (fid, sid) = self
            .framebuffer
            .get_frame_data(Some(frame as usize), Some(sem_index));

        let (framebuffer_fence, framebuffer, command_pool, command_buffers) = fid.unwrap();
        let (image_acquired, image_present) = sid.unwrap();

        unsafe {
            self.device
                .borrow()
                .device
                .wait_for_fence(framebuffer_fence, !0)
                .unwrap();
            self.device
                .borrow()
                .device
                .reset_fence(framebuffer_fence)
                .unwrap();
            command_pool.reset(false);

            // Rendering
            let mut cmd_buffer = match command_buffers.pop() {
                Some(cmd_buffer) => cmd_buffer,
                None => command_pool.allocate_one(command::Level::Primary),
            };
            cmd_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);

            cmd_buffer.set_viewports(0, &[self.viewport.clone()]);
            cmd_buffer.set_scissors(0, &[self.viewport.rect]);
            cmd_buffer.bind_graphics_pipeline(self.pipeline.pipeline.as_ref().unwrap());
            cmd_buffer.bind_vertex_buffers(0, Some((self.vertex_buffer.get_buffer(), 0)));
            cmd_buffer.bind_graphics_descriptor_sets(
                self.pipeline.pipeline_layout.as_ref().unwrap(),
                0,
                vec![
                    self.image.desc.set.as_ref().unwrap(),
                    self.uniform.desc.as_ref().unwrap().set.as_ref().unwrap(),
                ],
                &[],
            );

            cmd_buffer.begin_render_pass(
                self.render_pass.render_pass.as_ref().unwrap(),
                &framebuffer,
                self.viewport.rect,
                &[command::ClearValue {
                    color: command::ClearColor {
                        float32: [cr, cg, cb, 1.0],
                    },
                }],
                command::SubpassContents::Inline,
            );
            cmd_buffer.draw(0..6, 0..1);
            cmd_buffer.end_render_pass();
            cmd_buffer.finish();

            let submission = Submission {
                command_buffers: iter::once(&cmd_buffer),
                wait_semaphores: iter::once((&*image_acquired, PipelineStage::BOTTOM_OF_PIPE)),
                signal_semaphores: iter::once(&*image_present),
            };

            self.device.borrow_mut().queues.queues[0].submit(submission, Some(framebuffer_fence));
            command_buffers.push(cmd_buffer);

            // present frame
            match self
                .swapchain
                .as_ref()
                .unwrap()
                .swapchain
                .as_ref()
                .unwrap()
                .present(
                    &mut self.device.borrow_mut().queues.queues[0],
                    frame,
                    Some(&*image_present),
                ) {
                Ok(_) => Ok(()),
                Err(_) => Err("Failed to present"),
            }
        }
    }
}

impl<B: Backend> Drop for RendererState<B> {
    fn drop(&mut self) {
        self.device.borrow().device.wait_idle().unwrap();
        unsafe {
            self.device
                .borrow()
                .device
                .destroy_descriptor_pool(self.img_desc_pool.take().unwrap());
            self.device
                .borrow()
                .device
                .destroy_descriptor_pool(self.uniform_desc_pool.take().unwrap());
            self.swapchain.take();
        }
    }
}
