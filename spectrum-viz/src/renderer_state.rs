extern crate gfx_hal as hal;

use std::borrow;
use std::cell::RefCell;
use std::io::Cursor;
use std::iter;
use std::rc::Rc;
use std::time::Instant;

use hal::buffer;
use hal::command;
use hal::format as f;
use hal::image as i;
use hal::pool;
use hal::prelude::*;
use hal::pso;
use hal::pso::ShaderStageFlags;
use hal::queue::Submission;
use hal::window as w;
use hal::Backend;

use crate::backend_state::BackendState;
use crate::buffer_state::BufferState;
use crate::desc_set::DescSetLayout;
use crate::device_state::DeviceState;
use crate::framebuffer_state::FramebufferState;
use crate::gx_constant::{ACTUAL_QUAD, DIMS, QUAD, TRIANGLE};
use crate::gx_object::Vertex;
use crate::image_state::ImageState;
use crate::pipeline_state::PipelineState;
use crate::render_pass_state::RenderPassState;
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
    device: Rc<RefCell<DeviceState<B>>>,
}

fn create_viewport(extent: i::Extent) -> pso::Viewport {
    pso::Viewport {
        rect: pso::Rect {
            x: 0,
            y: 0,
            w: extent.width as i16,
            h: extent.height as i16,
        },
        depth: 0.0..1.0,
    }
}

impl<B: Backend> RendererState<B> {
    fn configure_swapchain(
        backend: &mut BackendState<B>,
        device: Rc<RefCell<DeviceState<B>>>,
    ) -> (f::Format, i::Extent) {
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
        println!("Swap config: {:?}", swap_config);
        let extent = swap_config.extent.to_extent();
        unsafe {
            backend
                .surface
                .configure_swapchain(&device.borrow().device, swap_config)
                .expect("Can't configure swapchain");
        };

        (format, extent)
    }

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
                    ty: pso::DescriptorType::Image {
                        ty: pso::ImageDescriptorType::Sampled {
                            with_sampler: false,
                        },
                    },
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
                ty: pso::DescriptorType::Buffer {
                    ty: pso::BufferDescriptorType::Uniform,
                    format: pso::BufferDescriptorFormat::Structured {
                        dynamic_offset: false,
                    },
                },
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
                        ty: pso::DescriptorType::Image {
                            ty: pso::ImageDescriptorType::Sampled {
                                with_sampler: false,
                            },
                        },
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
                    ty: pso::DescriptorType::Buffer {
                        ty: pso::BufferDescriptorType::Uniform,
                        format: pso::BufferDescriptorFormat::Structured {
                            dynamic_offset: false,
                        },
                    },
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

        let (format, extent) = RendererState::configure_swapchain(&mut backend, Rc::clone(&device));

        let render_pass = RenderPassState::new(format, Rc::clone(&device));

        let framebuffer = FramebufferState::new(Rc::clone(&device));

        let pipeline = PipelineState::new_triangle(
            vec![image.get_layout(), uniform.get_layout()],
            render_pass.render_pass.as_ref().unwrap(),
            Rc::clone(&device),
        );

        let viewport = create_viewport(extent);

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
            framebuffer,
            viewport,
            creation_instant: std::time::Instant::now(),
        }
    }

    pub fn recreate_swapchain(&mut self) {
        let (_, extent) =
            RendererState::configure_swapchain(&mut self.backend, Rc::clone(&self.device));
        self.viewport = create_viewport(extent);
    }

    pub fn draw_triangle(&mut self, cr: f32, cg: f32, cb: f32) -> Result<(), &'static str> {
        let surface_image = unsafe {
            match self.backend.surface.acquire_image(!0) {
                Ok((image, _)) => image,
                Err(_) => {
                    self.recreate_swapchain();
                    return Ok(());
                }
            }
        };

        let framebuffer = unsafe {
            self.device
                .borrow()
                .device
                .create_framebuffer(
                    self.render_pass.render_pass.as_ref().unwrap(),
                    iter::once(borrow::Borrow::borrow(&surface_image)),
                    i::Extent {
                        width: DIMS.width,
                        height: DIMS.height,
                        depth: 1,
                    },
                )
                .unwrap()
        };

        let (command_pool, cmd_buffer, present_fence, present_semaphore) =
            self.framebuffer.next_frame_info();

        let duration = std::time::Instant::now().duration_since(self.creation_instant);
        let time_f32 = duration.as_secs() as f32 + duration.subsec_nanos() as f32 * 1e-9;

        unsafe {
            self.device
                .borrow()
                .device
                .wait_for_fence(present_fence, !0)
                .unwrap();
            self.device
                .borrow()
                .device
                .reset_fence(present_fence)
                .unwrap();
            command_pool.reset(false);

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
                wait_semaphores: None,
                signal_semaphores: iter::once(&*present_semaphore),
            };

            self.device.borrow_mut().queues.queues[0].submit(submission, Some(present_fence));
            let result = self.device.borrow_mut().queues.queues[0].present_surface(
                &mut self.backend.surface,
                surface_image,
                Some(present_semaphore),
            );

            self.device.borrow().device.destroy_framebuffer(framebuffer);

            if let Err(e) = result {
                println!("Error when presenting: {}", e);
                self.recreate_swapchain();
            }
            Ok(())
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
            self.backend
                .surface
                .unconfigure_swapchain(&self.device.borrow().device);
        }
    }
}
