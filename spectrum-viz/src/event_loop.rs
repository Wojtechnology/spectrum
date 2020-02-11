#![cfg_attr(
    not(any(feature = "vulkan", feature = "metal",)),
    allow(dead_code, unused_extern_crates, unused_imports)
)]

use std::sync::{Arc, RwLock};

use env_logger;
#[cfg(not(any(feature = "vulkan", feature = "metal",)))]
use gfx_backend_empty as back;
#[cfg(feature = "metal")]
use gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as back;
use winit;

use crate::backend_state::create_backend;
use crate::renderer_state::{RendererState, UserState};
use crate::screen_size_state::ScreenSizeState;
use spectrum_audio::shared_data::SharedData;

fn init() -> (
    winit::event_loop::EventLoop<()>,
    RendererState<back::Backend>,
) {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new();
    let min_screen_state = ScreenSizeState::new_default_min(1.0);
    let start_screen_state = ScreenSizeState::new_default_start(1.0);
    let wb = winit::window::WindowBuilder::new()
        .with_min_inner_size(min_screen_state.logical_size())
        .with_inner_size(start_screen_state.logical_size())
        .with_title("colour-uniform".to_string());
    let backend = create_backend(wb, &event_loop);

    let renderer_state = unsafe { RendererState::new(backend) };
    (event_loop, renderer_state)
}

#[cfg(any(feature = "vulkan", feature = "metal",))]
pub fn run(shared_data: Arc<RwLock<SharedData>>) {
    let (event_loop, mut renderer_state) = init();
    let mut user_state = UserState::new(shared_data);

    renderer_state.draw(&user_state);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;
        match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
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
                winit::event::WindowEvent::Resized(size) => {
                    renderer_state.resize(size);
                }
                winit::event::WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                    renderer_state.change_dpi(scale_factor);
                }
                winit::event::WindowEvent::CursorMoved { position, .. } => {
                    user_state.cursor_pos = position;
                }
                _ => (),
            },
            winit::event::Event::RedrawRequested(_) => {
                renderer_state.draw(&user_state);
            }
            winit::event::Event::MainEventsCleared => {
                renderer_state.backend.window.request_redraw();
            }
            _ => (),
        };
    });
}

#[cfg(not(any(feature = "vulkan", feature = "metal",)))]
pub fn run() {
    println!("You need to enable the native API feature (vulkan/metal) in order to test the LL");
}
