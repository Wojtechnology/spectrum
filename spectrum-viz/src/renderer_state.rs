use std::borrow;
use std::cell::RefCell;
use std::iter;
use std::rc::Rc;
use std::sync::{Arc, RwLock};

use gfx_hal as hal;
use hal::buffer;
use hal::command;
use hal::format as f;
use hal::image as i;
use hal::prelude::*;
use hal::pso;
use hal::pso::ShaderStageFlags;
use hal::queue::Submission;
use hal::window as w;
use hal::Backend;
use hal::IndexType;
use winit::dpi::{PhysicalPosition, PhysicalSize};

use crate::backend_state::BackendState;
use crate::buffer_state::BufferState;
use crate::device_state::DeviceState;
use crate::framebuffer_state::FramebufferState;
use crate::gx_constant;
use crate::gx_object::{TriIndexData, VertexData};
use crate::pipeline_state::PipelineState;
use crate::render_pass_state::RenderPassState;
use crate::screen_size_state::ScreenSizeState;
use spectrum_audio::shared_data::SharedData;

// TODO: Move into own module
#[derive(Copy, Clone)]
pub struct Color<N> {
    pub r: N,
    pub g: N,
    pub b: N,
}

// TODO: Move into own module
#[derive(Clone)]
pub struct UserState {
    pub cursor_pos: PhysicalPosition<i32>,
    pub clear_color: Color<f32>,
    shared_data: Arc<RwLock<SharedData>>,
}

impl UserState {
    pub fn new(shared_data: Arc<RwLock<SharedData>>) -> Self {
        UserState {
            cursor_pos: PhysicalPosition::new(0, 0),
            clear_color: Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
            },
            shared_data,
        }
    }
}

pub struct RendererState<B: Backend> {
    pub viewport: pso::Viewport,
    // Locally defined data
    pub backend: BackendState<B>,
    render_pass: RenderPassState<B>,
    triangle_vertex_buffer: BufferState<B>,
    index_buffer: BufferState<B>,
    pipeline: PipelineState<B>,
    framebuffer: FramebufferState<B>,
    device: Rc<RefCell<DeviceState<B>>>,
    screen_size_state: ScreenSizeState,
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
        screen_size_state: &ScreenSizeState,
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
        let swap_config =
            w::SwapchainConfig::from_caps(&caps, format, screen_size_state.extent_2d());
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

        println!("Memory types: {:?}", backend.adapter.memory_types);

        let triangle_vertex_buffer = BufferState::new::<VertexData<f32>>(
            Rc::clone(&device),
            &gx_constant::triangle_vertices(),
            buffer::Usage::VERTEX,
            &backend.adapter.memory_types,
        );

        let index_buffer = BufferState::new::<TriIndexData<u16>>(
            Rc::clone(&device),
            &gx_constant::triangle_indices(),
            buffer::Usage::INDEX,
            &backend.adapter.memory_types,
        );

        let screen_size_state = ScreenSizeState::new_default_start(backend.window.scale_factor());
        let (format, extent) =
            Self::configure_swapchain(&mut backend, Rc::clone(&device), &screen_size_state);

        let render_pass = RenderPassState::new(format, Rc::clone(&device));

        let framebuffer = FramebufferState::new(Rc::clone(&device));

        let pipeline = PipelineState::new(
            Vec::<B::DescriptorSetLayout>::new(),
            render_pass.render_pass.as_ref().unwrap(),
            Rc::clone(&device),
        );

        let viewport = create_viewport(extent);

        RendererState {
            backend,
            device,
            triangle_vertex_buffer,
            index_buffer,
            render_pass,
            pipeline,
            framebuffer,
            viewport,
            screen_size_state,
        }
    }

    fn recreate_swapchain(&mut self) {
        let (_, extent) = Self::configure_swapchain(
            &mut self.backend,
            Rc::clone(&self.device),
            &self.screen_size_state,
        );
        self.viewport = create_viewport(extent);
    }

    pub fn draw(&mut self, user_state: &UserState) {
        let surface_image = unsafe {
            match self.backend.surface.acquire_image(!0) {
                Ok((image, _)) => image,
                Err(_) => {
                    self.recreate_swapchain();
                    return;
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
                    self.screen_size_state.extent(),
                )
                .unwrap()
        };

        let (command_pool, cmd_buffer, present_fence, present_semaphore) =
            self.framebuffer.next_frame_info();

        let sample_vec = user_state.shared_data.read().unwrap().get_sample();
        let value = sample_vec[0];
        let value_f = ((value as f32) / (i16::max_value() as f32)).abs();

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
            cmd_buffer.bind_index_buffer(buffer::IndexBufferView {
                buffer: self.index_buffer.get_buffer(),
                offset: 0,
                index_type: IndexType::U16,
            });
            cmd_buffer.push_graphics_constants(
                self.pipeline.pipeline_layout.as_ref().unwrap(),
                ShaderStageFlags::FRAGMENT | ShaderStageFlags::VERTEX,
                0,
                &[value_f.to_bits()],
            );

            cmd_buffer.begin_render_pass(
                self.render_pass.render_pass.as_ref().unwrap(),
                &framebuffer,
                self.viewport.rect,
                &[command::ClearValue {
                    color: command::ClearColor {
                        float32: [
                            user_state.clear_color.r,
                            user_state.clear_color.g,
                            user_state.clear_color.b,
                            1.0,
                        ],
                    },
                }],
                command::SubpassContents::Inline,
            );
            cmd_buffer.draw_indexed(0..12, 0, 0..1);
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
        }
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        self.screen_size_state.set_physical_size(size);
        self.recreate_swapchain();
    }

    pub fn change_dpi(&mut self, dpi_factor: f64) {
        self.screen_size_state.set_dpi_factor(dpi_factor);
        self.recreate_swapchain();
    }
}

impl<B: Backend> Drop for RendererState<B> {
    fn drop(&mut self) {
        self.device.borrow().device.wait_idle().unwrap();
        unsafe {
            self.backend
                .surface
                .unconfigure_swapchain(&self.device.borrow().device);
        }
    }
}
