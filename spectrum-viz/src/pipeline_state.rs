use std::cell::RefCell;
use std::fs;
use std::mem::size_of;
use std::rc::Rc;

use gfx_hal as hal;
use hal::format as f;
use hal::pass::Subpass;
use hal::prelude::*;
use hal::pso;
use hal::pso::{ShaderStageFlags, VertexInputRate};
use hal::Backend;

use crate::device_state::DeviceState;
use crate::gx_object::VertexData;

const ENTRY_NAME: &str = "main";

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

pub struct PipelineState<B: Backend> {
    pub pipeline: Option<B::GraphicsPipeline>,
    pub pipeline_layout: Option<B::PipelineLayout>,
    device: Rc<RefCell<DeviceState<B>>>,
}

impl<B: Backend> PipelineState<B> {
    pub unsafe fn new<IS>(
        desc_layouts: IS,
        render_pass: &B::RenderPass,
        device_ptr: Rc<RefCell<DeviceState<B>>>,
    ) -> Self
    where
        IS: IntoIterator,
        IS::Item: std::borrow::Borrow<B::DescriptorSetLayout>,
    {
        let device = &device_ptr.borrow().device;
        let push_constants = vec![
            (ShaderStageFlags::FRAGMENT, 0..4),
            (ShaderStageFlags::VERTEX, 0..4),
        ];
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
                    stride: size_of::<VertexData<f32>>() as u32,
                    rate: VertexInputRate::Vertex,
                });

                pipeline_desc.attributes.push(pso::AttributeDesc {
                    location: 0,
                    binding: 0,
                    element: pso::Element {
                        format: f::Format::Rgb32Sfloat,
                        offset: 0,
                    },
                });
                pipeline_desc.attributes.push(pso::AttributeDesc {
                    location: 1,
                    binding: 0,
                    element: pso::Element {
                        format: f::Format::Rgb32Sfloat,
                        offset: size_of::<[f32; 3]>() as u32,
                    },
                });
                pipeline_desc.attributes.push(pso::AttributeDesc {
                    location: 2,
                    binding: 0,
                    element: pso::Element {
                        format: f::Format::Rg32Sfloat,
                        offset: size_of::<[f32; 6]>() as u32,
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
