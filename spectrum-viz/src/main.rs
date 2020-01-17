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

extern crate env_logger;
extern crate gfx_hal as hal;
extern crate glsl_to_spirv;
extern crate image;
extern crate log;
extern crate winit;

use spectrum_viz::backend_state::create_backend;
use spectrum_viz::renderer_state::{RendererState, UserState};
use spectrum_viz::screen_size_state::ScreenSizeState;

fn init() -> (
    winit::event_loop::EventLoop<()>,
    RendererState<back::Backend>,
) {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new();
    let min_screen_state = ScreenSizeState::new_default_min();
    let start_screen_state = ScreenSizeState::new_default_start();
    let wb = winit::window::WindowBuilder::new()
        .with_min_inner_size(min_screen_state.logical_size())
        .with_inner_size(start_screen_state.logical_size())
        .with_title("colour-uniform".to_string());
    let backend = create_backend(wb, &event_loop);

    let renderer_state = unsafe { RendererState::new(backend) };
    (event_loop, renderer_state)
}

#[cfg(any(feature = "vulkan", feature = "metal",))]
fn main() {
    let (event_loop, mut renderer_state) = init();
    let mut user_state = UserState::new();

    renderer_state.draw(user_state);

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
                winit::event::WindowEvent::HiDpiFactorChanged(dpi_factor) => {
                    renderer_state.change_dpi(dpi_factor);
                }
                winit::event::WindowEvent::RedrawRequested => {
                    renderer_state.draw(user_state);
                }
                winit::event::WindowEvent::CursorMoved { position, .. } => {
                    user_state.cursor_pos = position;
                }
                _ => (),
            },
            winit::event::Event::EventsCleared => {
                renderer_state.backend.window.request_redraw();
            }
            _ => (),
        };
    });
}

#[cfg(not(any(feature = "vulkan", feature = "metal",)))]
fn main() {
    println!("You need to enable the native API feature (vulkan/metal) in order to test the LL");
}
