use std::borrow;
use std::cell::RefCell;
use std::iter;
use std::rc::Rc;
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

use gfx::memory::cast_slice;
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
use nalgebra_glm as glm;
use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize};

use crate::backend_state::BackendState;
use crate::buffer_state::BufferState;
use crate::device_state::DeviceState;
use crate::framebuffer_state::{FramebufferState, MAX_FRAMES_IN_FLIGHT};
use crate::gx_constant;
use crate::gx_object::{TriIndexData, VertexData};
use crate::pipeline_state::PipelineState;
use crate::render_pass_state::RenderPassState;
use crate::screen_size_state::ScreenSizeState;
use spectrum_audio::shared_data::SharedData;

pub const MAX_CUBES: usize = 8192;
pub const DISPLAY_WIDTH: f32 = 5.5;
pub const VERT_SCALE: f32 = 0.25;

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
    start_time: SystemTime,
    shared_data: Arc<RwLock<SharedData>>,
}

fn mat4_to_array(mat: glm::Mat4) -> [f32; 16] {
    let mut arr = [0.0; 16];
    for i in 0..4 {
        for j in 0..4 {
            arr[i * 4 + j] = mat[i * 4 + j];
        }
    }
    arr
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
            start_time: SystemTime::now(),
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
    model_vertex_buffers: Vec<BufferState<B>>,
    index_buffer: BufferState<B>,
    pipeline: PipelineState<B>,
    framebuffer: FramebufferState<B>,
    device: Rc<RefCell<DeviceState<B>>>,
    screen_size_state: ScreenSizeState,
    view_proj_mat: glm::Mat4,
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
            &gx_constant::cube_vertices(),
            buffer::Usage::VERTEX,
            &backend.adapter.memory_types,
        );

        let index_buffer = BufferState::new::<TriIndexData<u16>>(
            Rc::clone(&device),
            &gx_constant::cube_indices(),
            buffer::Usage::INDEX,
            &backend.adapter.memory_types,
        );

        // TODO: Clean up buffer_state to separate uploading and creation
        let mut model_vertex_buffers = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            model_vertex_buffers.push(BufferState::new::<[f32; 4 * 4]>(
                Rc::clone(&device),
                &[mat4_to_array(glm::TMat4::<f32>::identity()); MAX_CUBES][..],
                buffer::Usage::VERTEX,
                &backend.adapter.memory_types,
            ));
        }

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
        let view_proj_mat = Self::compute_view_proj(screen_size_state.logical_size());

        RendererState {
            backend,
            device,
            triangle_vertex_buffer,
            model_vertex_buffers,
            index_buffer,
            render_pass,
            pipeline,
            framebuffer,
            viewport,
            screen_size_state,
            view_proj_mat,
        }
    }

    fn compute_view_proj(size: LogicalSize<u32>) -> glm::Mat4 {
        let view = glm::look_at_lh(
            &glm::make_vec3(&[0.0, 0.0, -5.0]),
            &glm::make_vec3(&[0.0, 0.0, 0.0]),
            &glm::make_vec3(&[0.0, 1.0, 0.0]).normalize(),
        );
        let projection = {
            let mut temp = glm::perspective_lh_zo(
                (size.width as f32) / (size.height as f32),
                f32::to_radians(50.0),
                0.1,
                100.0,
            );
            temp[(1, 1)] *= -1.0;
            temp
        };
        projection * view
    }

    fn recreate_swapchain(&mut self) {
        let (_, extent) = Self::configure_swapchain(
            &mut self.backend,
            Rc::clone(&self.device),
            &self.screen_size_state,
        );
        self.viewport = create_viewport(extent);
        self.view_proj_mat = Self::compute_view_proj(self.screen_size_state.logical_size());
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

        let (frame_idx, command_pool, cmd_buffer, present_fence, present_semaphore) =
            self.framebuffer.next_frame_info();

        let bands = user_state.shared_data.read().unwrap().get_bands();

        // SCALE
        let mut models = Vec::new();
        let bar_width = DISPLAY_WIDTH / (bands.len() as f32);
        let mut max_val = 0.0;
        for (model_idx, &avg_value) in bands.iter().enumerate() {
            let x_translate = bar_width * (model_idx as f32) - DISPLAY_WIDTH / 2.0;
            if avg_value > max_val {
                max_val = avg_value;
            }

            let model = {
                let mut model = glm::TMat4::<f32>::identity();
                model = glm::translate(&model, &glm::TVec3::new(x_translate, -1.8, 0.0));
                model = glm::scale(
                    &model,
                    &glm::TVec3::new(bar_width, avg_value / VERT_SCALE, 1.0),
                );
                model = glm::translate(&model, &glm::TVec3::new(-0.5, 0.0, -0.5));
                model
            };
            models.push(model);
        }
        let model_arrays: Vec<_> = models.iter().map(|&mat| mat4_to_array(mat)).collect();

        let model_vertex_buffer = &mut self.model_vertex_buffers[frame_idx];
        model_vertex_buffer.update_data(0, &model_arrays[..]);

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
            cmd_buffer.bind_vertex_buffers(
                0,
                vec![
                    (self.triangle_vertex_buffer.get_buffer(), 0),
                    (model_vertex_buffer.get_buffer(), 0),
                ],
            );
            cmd_buffer.bind_index_buffer(buffer::IndexBufferView {
                buffer: self.index_buffer.get_buffer(),
                offset: 0,
                index_type: IndexType::U16,
            });
            cmd_buffer.push_graphics_constants(
                self.pipeline.pipeline_layout.as_ref().unwrap(),
                ShaderStageFlags::VERTEX,
                0,
                cast_slice::<f32, u32>(&self.view_proj_mat.data),
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
            cmd_buffer.draw_indexed(0..36, 0, 0..(bands.len() as u32));
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
