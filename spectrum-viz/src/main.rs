#![cfg_attr(
    not(any(feature = "vulkan", feature = "metal",)),
    allow(dead_code, unused_extern_crates, unused_imports)
)]

#[cfg(not(any(feature = "vulkan", feature = "metal",)))]
extern crate gfx_backend_empty as back;
#[cfg(feature = "metal")]
extern crate gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as back;

#[macro_use]
extern crate log;
extern crate env_logger;
extern crate gfx_hal as hal;
extern crate glsl_to_spirv;
extern crate image;
extern crate winit;

use std::cell::RefCell;
use std::io::Cursor;
use std::mem::size_of;
use std::rc::Rc;
use std::time::Instant;
use std::{fs, iter};

use hal::format::{Rgba8Srgb as ColorFormat, Swizzle};
use hal::pass::Subpass;
use hal::pso::{PipelineStage, ShaderStageFlags, VertexInputRate};
use hal::{
    buffer, command,
    format::{self as f, AsFormat},
    image as i, memory as m, pool,
    prelude::*,
    pso,
    queue::Submission,
    window as w, Backend,
};

use spectrum_viz::adapter_state::AdapterState;
use spectrum_viz::backend_state::{create_backend, BackendState};
use spectrum_viz::buffer_state::BufferState;
use spectrum_viz::desc_set::{DescSet, DescSetLayout, DescSetWrite};
use spectrum_viz::device_state::DeviceState;
use spectrum_viz::framebuffer_state::FramebufferState;
use spectrum_viz::gx_constant::{ACTUAL_QUAD, COLOR_RANGE, DIMS, QUAD, TRIANGLE};
use spectrum_viz::gx_object::Vertex;
use spectrum_viz::render_pass_state::RenderPassState;
use spectrum_viz::swapchain_state::SwapchainState;
use spectrum_viz::uniform::Uniform;

const ENTRY_NAME: &str = "main";

struct RendererState<B: Backend> {
    uniform_desc_pool: Option<B::DescriptorPool>,
    img_desc_pool: Option<B::DescriptorPool>,
    viewport: pso::Viewport,
    creation_instant: Instant,
    // Locally defined data
    backend: BackendState<B>,
    render_pass: RenderPassState<B>,
    vertex_buffer: BufferState<B>,
    triangle_vertex_buffer: BufferState<B>,
    quad_vertex_buffer: BufferState<B>,
    image: ImageState<B>,
    pipeline: PipelineState<B>,
    framebuffer: FramebufferState<B>,
    uniform: Uniform<B>,
    swapchain: Option<SwapchainState<B>>,
    device: Rc<RefCell<DeviceState<B>>>,
}

#[derive(Debug)]
enum Color {
    Red,
    Green,
    Blue,
    Alpha,
}

impl<B: Backend> RendererState<B> {
    unsafe fn new(mut backend: BackendState<B>) -> Self {
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

    fn recreate_swapchain(&mut self) {
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

    fn draw_triangle(&mut self, cr: f32, cg: f32, cb: f32) -> Result<(), &'static str> {
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

struct ImageState<B: Backend> {
    desc: DescSet<B>,
    buffer: Option<BufferState<B>>,
    sampler: Option<B::Sampler>,
    image_view: Option<B::ImageView>,
    image: Option<B::Image>,
    memory: Option<B::Memory>,
    transfered_image_fence: Option<B::Fence>,
}

impl<B: Backend> ImageState<B> {
    unsafe fn new(
        mut desc: DescSet<B>,
        img: &::image::ImageBuffer<::image::Rgba<u8>, Vec<u8>>,
        adapter: &AdapterState<B>,
        usage: buffer::Usage,
        device_state: &mut DeviceState<B>,
        staging_pool: &mut B::CommandPool,
    ) -> Self {
        let (buffer, dims, row_pitch, stride) = BufferState::new_texture(
            Rc::clone(&desc.layout.device),
            &mut device_state.device,
            img,
            adapter,
            usage,
        );

        let buffer = Some(buffer);
        let device = &mut device_state.device;

        let kind = i::Kind::D2(dims.width as i::Size, dims.height as i::Size, 1, 1);
        let mut image = device
            .create_image(
                kind,
                1,
                ColorFormat::SELF,
                i::Tiling::Optimal,
                i::Usage::TRANSFER_DST | i::Usage::SAMPLED,
                i::ViewCapabilities::empty(),
            )
            .unwrap(); // TODO: usage
        let req = device.get_image_requirements(&image);

        let device_type = adapter
            .memory_types
            .iter()
            .enumerate()
            .position(|(id, memory_type)| {
                req.type_mask & (1 << id) != 0
                    && memory_type.properties.contains(m::Properties::DEVICE_LOCAL)
            })
            .unwrap()
            .into();

        let memory = device.allocate_memory(device_type, req.size).unwrap();

        device.bind_image_memory(&memory, 0, &mut image).unwrap();
        let image_view = device
            .create_image_view(
                &image,
                i::ViewKind::D2,
                ColorFormat::SELF,
                Swizzle::NO,
                COLOR_RANGE.clone(),
            )
            .unwrap();

        let sampler = device
            .create_sampler(&i::SamplerDesc::new(i::Filter::Linear, i::WrapMode::Clamp))
            .expect("Can't create sampler");

        desc.write_to_state(
            vec![
                DescSetWrite {
                    binding: 0,
                    array_offset: 0,
                    descriptors: Some(pso::Descriptor::Image(
                        &image_view,
                        i::Layout::ShaderReadOnlyOptimal,
                    )),
                },
                DescSetWrite {
                    binding: 1,
                    array_offset: 0,
                    descriptors: Some(pso::Descriptor::Sampler(&sampler)),
                },
            ],
            device,
        );

        let mut transfered_image_fence = device.create_fence(false).expect("Can't create fence");

        // copy buffer to texture
        {
            let mut cmd_buffer = staging_pool.allocate_one(command::Level::Primary);
            cmd_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);

            let image_barrier = m::Barrier::Image {
                states: (i::Access::empty(), i::Layout::Undefined)
                    ..(i::Access::TRANSFER_WRITE, i::Layout::TransferDstOptimal),
                target: &image,
                families: None,
                range: COLOR_RANGE.clone(),
            };

            cmd_buffer.pipeline_barrier(
                PipelineStage::TOP_OF_PIPE..PipelineStage::TRANSFER,
                m::Dependencies::empty(),
                &[image_barrier],
            );

            cmd_buffer.copy_buffer_to_image(
                buffer.as_ref().unwrap().get_buffer(),
                &image,
                i::Layout::TransferDstOptimal,
                &[command::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_width: row_pitch / (stride as u32),
                    buffer_height: dims.height as u32,
                    image_layers: i::SubresourceLayers {
                        aspects: f::Aspects::COLOR,
                        level: 0,
                        layers: 0..1,
                    },
                    image_offset: i::Offset { x: 0, y: 0, z: 0 },
                    image_extent: i::Extent {
                        width: dims.width,
                        height: dims.height,
                        depth: 1,
                    },
                }],
            );

            let image_barrier = m::Barrier::Image {
                states: (i::Access::TRANSFER_WRITE, i::Layout::TransferDstOptimal)
                    ..(i::Access::SHADER_READ, i::Layout::ShaderReadOnlyOptimal),
                target: &image,
                families: None,
                range: COLOR_RANGE.clone(),
            };
            cmd_buffer.pipeline_barrier(
                PipelineStage::TRANSFER..PipelineStage::FRAGMENT_SHADER,
                m::Dependencies::empty(),
                &[image_barrier],
            );

            cmd_buffer.finish();

            device_state.queues.queues[0].submit_without_semaphores(
                iter::once(&cmd_buffer),
                Some(&mut transfered_image_fence),
            );
        }

        ImageState {
            desc: desc,
            buffer: buffer,
            sampler: Some(sampler),
            image_view: Some(image_view),
            image: Some(image),
            memory: Some(memory),
            transfered_image_fence: Some(transfered_image_fence),
        }
    }

    fn wait_for_transfer_completion(&self) {
        let device = &self.desc.layout.device.borrow().device;
        unsafe {
            device
                .wait_for_fence(self.transfered_image_fence.as_ref().unwrap(), !0)
                .unwrap();
        }
    }

    fn get_layout(&self) -> &B::DescriptorSetLayout {
        self.desc.get_layout()
    }
}

impl<B: Backend> Drop for ImageState<B> {
    fn drop(&mut self) {
        unsafe {
            let device = &self.desc.layout.device.borrow().device;

            let fence = self.transfered_image_fence.take().unwrap();
            device.wait_for_fence(&fence, !0).unwrap();
            device.destroy_fence(fence);

            device.destroy_sampler(self.sampler.take().unwrap());
            device.destroy_image_view(self.image_view.take().unwrap());
            device.destroy_image(self.image.take().unwrap());
            device.free_memory(self.memory.take().unwrap());
        }

        self.buffer.take().unwrap();
    }
}

unsafe fn read_shader_str<B: Backend>(
    device: &B::Device,
    shader_str: &str,
    shader_type: glsl_to_spirv::ShaderType,
) -> B::ShaderModule {
    let file = glsl_to_spirv::compile(shader_str, shader_type).unwrap();
    let spirv: Vec<u32> = pso::read_spirv(file).unwrap();
    device.create_shader_module(&spirv).unwrap()
}

unsafe fn read_shader_file<B: Backend>(
    device: &B::Device,
    file_path: &str,
    shader_type: glsl_to_spirv::ShaderType,
) -> B::ShaderModule {
    let glsl = fs::read_to_string(file_path).unwrap();
    read_shader_str::<B>(device, &glsl, shader_type)
}

struct PipelineState<B: Backend> {
    pipeline: Option<B::GraphicsPipeline>,
    pipeline_layout: Option<B::PipelineLayout>,
    device: Rc<RefCell<DeviceState<B>>>,
}

impl<B: Backend> PipelineState<B> {
    unsafe fn new_triangle<IS>(
        desc_layouts: IS,
        render_pass: &B::RenderPass,
        device_ptr: Rc<RefCell<DeviceState<B>>>,
    ) -> Self
    where
        IS: IntoIterator,
        IS::Item: std::borrow::Borrow<B::DescriptorSetLayout>,
    {
        let device = &device_ptr.borrow().device;
        let push_constants = vec![(ShaderStageFlags::FRAGMENT, 0..4)];
        let pipeline_layout = device
            .create_pipeline_layout(desc_layouts, push_constants)
            .expect("Can't create pipeline layout");

        let vs_module = read_shader_file::<B>(
            device,
            "src/data/tri.vert",
            glsl_to_spirv::ShaderType::Vertex,
        );
        let fs_module = read_shader_file::<B>(
            device,
            "src/data/tri.frag",
            glsl_to_spirv::ShaderType::Fragment,
        );

        let pipeline = {
            let pipeline = {
                let (vs_entry, fs_entry) = (
                    pso::EntryPoint::<B> {
                        entry: ENTRY_NAME,
                        module: &vs_module,
                        specialization: pso::Specialization::default(),
                    },
                    pso::EntryPoint::<B> {
                        entry: ENTRY_NAME,
                        module: &fs_module,
                        specialization: pso::Specialization::default(),
                    },
                );

                let shader_entries = pso::GraphicsShaderSet {
                    vertex: vs_entry,
                    hull: None,
                    domain: None,
                    geometry: None,
                    fragment: Some(fs_entry),
                };

                let subpass = Subpass {
                    index: 0,
                    main_pass: render_pass,
                };

                let mut pipeline_desc = pso::GraphicsPipelineDesc::new(
                    shader_entries,
                    pso::Primitive::TriangleList,
                    pso::Rasterizer::FILL,
                    &pipeline_layout,
                    subpass,
                );
                pipeline_desc.blender.targets.push(pso::ColorBlendDesc {
                    mask: pso::ColorMask::ALL,
                    blend: Some(pso::BlendState::ALPHA),
                });
                pipeline_desc.vertex_buffers.push(pso::VertexBufferDesc {
                    binding: 0,
                    stride: size_of::<[f32; 5]>() as u32,
                    rate: VertexInputRate::Vertex,
                });

                pipeline_desc.attributes.push(pso::AttributeDesc {
                    location: 0,
                    binding: 0,
                    element: pso::Element {
                        format: f::Format::Rg32Sfloat,
                        offset: 0,
                    },
                });
                pipeline_desc.attributes.push(pso::AttributeDesc {
                    location: 1,
                    binding: 0,
                    element: pso::Element {
                        format: f::Format::Rgb32Sfloat,
                        offset: size_of::<[f32; 2]>() as u32,
                    },
                });

                device.create_graphics_pipeline(&pipeline_desc, None)
            };

            device.destroy_shader_module(vs_module);
            device.destroy_shader_module(fs_module);

            pipeline.unwrap()
        };
        PipelineState {
            pipeline: Some(pipeline),
            pipeline_layout: Some(pipeline_layout),
            device: Rc::clone(&device_ptr),
        }
    }

    unsafe fn new<IS>(
        desc_layouts: IS,
        render_pass: &B::RenderPass,
        device_ptr: Rc<RefCell<DeviceState<B>>>,
    ) -> Self
    where
        IS: IntoIterator,
        IS::Item: std::borrow::Borrow<B::DescriptorSetLayout>,
    {
        let device = &device_ptr.borrow().device;
        let pipeline_layout = device
            .create_pipeline_layout(desc_layouts, &[(pso::ShaderStageFlags::VERTEX, 0..8)])
            .expect("Can't create pipeline layout");

        let pipeline = {
            let vs_module = read_shader_file::<B>(
                device,
                "src/data/quad.vert",
                glsl_to_spirv::ShaderType::Vertex,
            );
            let fs_module = read_shader_file::<B>(
                device,
                "src/data/quad.frag",
                glsl_to_spirv::ShaderType::Fragment,
            );

            let pipeline = {
                let (vs_entry, fs_entry) = (
                    pso::EntryPoint::<B> {
                        entry: ENTRY_NAME,
                        module: &vs_module,
                        specialization: hal::spec_const_list![0.8f32],
                    },
                    pso::EntryPoint::<B> {
                        entry: ENTRY_NAME,
                        module: &fs_module,
                        specialization: pso::Specialization::default(),
                    },
                );

                let shader_entries = pso::GraphicsShaderSet {
                    vertex: vs_entry,
                    hull: None,
                    domain: None,
                    geometry: None,
                    fragment: Some(fs_entry),
                };

                let subpass = Subpass {
                    index: 0,
                    main_pass: render_pass,
                };

                let mut pipeline_desc = pso::GraphicsPipelineDesc::new(
                    shader_entries,
                    pso::Primitive::TriangleList,
                    pso::Rasterizer::FILL,
                    &pipeline_layout,
                    subpass,
                );
                pipeline_desc.blender.targets.push(pso::ColorBlendDesc {
                    mask: pso::ColorMask::ALL,
                    blend: Some(pso::BlendState::ALPHA),
                });
                pipeline_desc.vertex_buffers.push(pso::VertexBufferDesc {
                    binding: 0,
                    stride: size_of::<Vertex>() as u32,
                    rate: VertexInputRate::Vertex,
                });

                pipeline_desc.attributes.push(pso::AttributeDesc {
                    location: 0,
                    binding: 0,
                    element: pso::Element {
                        format: f::Format::Rg32Sfloat,
                        offset: 0,
                    },
                });
                pipeline_desc.attributes.push(pso::AttributeDesc {
                    location: 1,
                    binding: 0,
                    element: pso::Element {
                        format: f::Format::Rg32Sfloat,
                        offset: 8,
                    },
                });

                device.create_graphics_pipeline(&pipeline_desc, None)
            };

            device.destroy_shader_module(vs_module);
            device.destroy_shader_module(fs_module);

            pipeline.unwrap()
        };

        PipelineState {
            pipeline: Some(pipeline),
            pipeline_layout: Some(pipeline_layout),
            device: Rc::clone(&device_ptr),
        }
    }
}

impl<B: Backend> Drop for PipelineState<B> {
    fn drop(&mut self) {
        let device = &self.device.borrow().device;
        unsafe {
            device.destroy_graphics_pipeline(self.pipeline.take().unwrap());
            device.destroy_pipeline_layout(self.pipeline_layout.take().unwrap());
        }
    }
}

#[cfg(any(feature = "vulkan", feature = "metal",))]
fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new();
    let wb = winit::window::WindowBuilder::new()
        .with_min_inner_size(winit::dpi::LogicalSize::new(1.0, 1.0))
        .with_inner_size(winit::dpi::LogicalSize::new(
            DIMS.width as _,
            DIMS.height as _,
        ))
        .with_title("colour-uniform".to_string());
    let backend = create_backend(wb, &event_loop);

    let mut renderer_state = unsafe { RendererState::new(backend) };
    let mut recreate_swapchain = false;

    let mut r = 1.0f32;
    let mut g = 1.0f32;
    let mut b = 1.0f32;
    let mut a = 1.0f32;

    let mut cr = 0.8;
    let mut cg = 0.8;
    let mut cb = 0.8;

    let mut x = 0.5;
    let mut y = 0.5;

    let mut cur_color = Color::Red;
    let mut cur_value: u32 = 0;

    println!("\nInstructions:");
    println!("\tChoose whether to change the (R)ed, (G)reen or (B)lue color by pressing the appropriate key.");
    println!("\tType in the value you want to change it to, where 0 is nothing, 255 is normal and 510 is double, ect.");
    println!("\tThen press C to change the (C)lear colour or (Enter) for the image color.");
    println!(
        "\tSet {:?} color to: {} (press enter/C to confirm)",
        cur_color, cur_value
    );

    match renderer_state.draw_triangle(cr, cg, cb) {
        Ok(()) => (),
        Err(_) => {
            recreate_swapchain = true;
        }
    }

    let mut frame_num: u64 = 0;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;
        let uniform = &mut renderer_state.uniform;
        let viewport = &mut renderer_state.viewport;
        match event {
            winit::event::Event::WindowEvent { event, .. } =>
            {
                #[allow(unused_variables)]
                match event {
                    winit::event::WindowEvent::KeyboardInput {
                        input:
                            winit::event::KeyboardInput {
                                virtual_keycode: Some(winit::event::VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    }
                    | winit::event::WindowEvent::CloseRequested => {
                        *control_flow = winit::event_loop::ControlFlow::Exit
                    }
                    winit::event::WindowEvent::Resized(dims) => {
                        recreate_swapchain = true;
                    }
                    winit::event::WindowEvent::RedrawRequested => {
                        if recreate_swapchain {
                            renderer_state.recreate_swapchain();
                            recreate_swapchain = false;
                        }

                        frame_num += 1;
                        println!("Frame num {} rendering!", frame_num);
                        match renderer_state.draw_triangle(cr, cg, cb) {
                            Ok(()) => (),
                            Err(e) => {
                                println!("Error: {}, recreating swapchain", e);
                                recreate_swapchain = true;
                            }
                        }
                    }
                    winit::event::WindowEvent::KeyboardInput {
                        input:
                            winit::event::KeyboardInput {
                                virtual_keycode,
                                state: winit::event::ElementState::Pressed,
                                ..
                            },
                        ..
                    } => {
                        if let Some(kc) = virtual_keycode {
                            match kc {
                                winit::event::VirtualKeyCode::Key0 => {
                                    cur_value = cur_value * 10 + 0
                                }
                                winit::event::VirtualKeyCode::Key1 => {
                                    cur_value = cur_value * 10 + 1
                                }
                                winit::event::VirtualKeyCode::Key2 => {
                                    cur_value = cur_value * 10 + 2
                                }
                                winit::event::VirtualKeyCode::Key3 => {
                                    cur_value = cur_value * 10 + 3
                                }
                                winit::event::VirtualKeyCode::Key4 => {
                                    cur_value = cur_value * 10 + 4
                                }
                                winit::event::VirtualKeyCode::Key5 => {
                                    cur_value = cur_value * 10 + 5
                                }
                                winit::event::VirtualKeyCode::Key6 => {
                                    cur_value = cur_value * 10 + 6
                                }
                                winit::event::VirtualKeyCode::Key7 => {
                                    cur_value = cur_value * 10 + 7
                                }
                                winit::event::VirtualKeyCode::Key8 => {
                                    cur_value = cur_value * 10 + 8
                                }
                                winit::event::VirtualKeyCode::Key9 => {
                                    cur_value = cur_value * 10 + 9
                                }
                                winit::event::VirtualKeyCode::R => {
                                    cur_value = 0;
                                    cur_color = Color::Red
                                }
                                winit::event::VirtualKeyCode::G => {
                                    cur_value = 0;
                                    cur_color = Color::Green
                                }
                                winit::event::VirtualKeyCode::B => {
                                    cur_value = 0;
                                    cur_color = Color::Blue
                                }
                                winit::event::VirtualKeyCode::A => {
                                    cur_value = 0;
                                    cur_color = Color::Alpha
                                }
                                winit::event::VirtualKeyCode::Return => {
                                    match cur_color {
                                        Color::Red => r = cur_value as f32 / 255.0,
                                        Color::Green => g = cur_value as f32 / 255.0,
                                        Color::Blue => b = cur_value as f32 / 255.0,
                                        Color::Alpha => a = cur_value as f32 / 255.0,
                                    }
                                    uniform
                                        .buffer
                                        .as_mut()
                                        .unwrap()
                                        .update_data(0, &[r, g, b, a]);
                                    cur_value = 0;

                                    println!("Colour updated!");
                                }
                                winit::event::VirtualKeyCode::C => {
                                    match cur_color {
                                        Color::Red => cr = cur_value as f32 / 255.0,
                                        Color::Green => cg = cur_value as f32 / 255.0,
                                        Color::Blue => cb = cur_value as f32 / 255.0,
                                        Color::Alpha => {
                                            error!("Alpha is not valid for the background.");
                                            return;
                                        }
                                    }
                                    cur_value = 0;

                                    println!("Background color updated!");
                                }
                                _ => return,
                            }
                            println!(
                                "Set {:?} color to: {} (press enter/C to confirm)",
                                cur_color, cur_value
                            )
                        }
                    }
                    winit::event::WindowEvent::CursorMoved { position, .. } => {
                        x = (position.x as f32) / (viewport.rect.w as f32) * 4.0 - 1.0;
                        y = (position.y as f32) / (viewport.rect.h as f32) * 4.0 - 1.0;
                    }
                    _ => (),
                }
            }
            winit::event::Event::EventsCleared => {
                renderer_state.backend.window.request_redraw();
                println!("EventsCleared");
            }
            _ => (),
        };
    });
}

#[cfg(not(any(feature = "vulkan", feature = "metal",)))]
fn main() {
    println!("You need to enable the native API feature (vulkan/metal) in order to test the LL");
}
