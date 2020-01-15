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

use spectrum_viz::backend_state::create_backend;
use spectrum_viz::gx_constant::DIMS;
use spectrum_viz::renderer_state::RendererState;

#[derive(Debug)]
enum Color {
    Red,
    Green,
    Blue,
    Alpha,
}

#[cfg(any(feature = "vulkan", feature = "metal",))]
fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new();
    let wb = winit::window::WindowBuilder::new()
        .with_min_inner_size(winit::dpi::LogicalSize::new(64.0, 64.0))
        .with_inner_size(winit::dpi::LogicalSize::new(
            DIMS.width as _,
            DIMS.height as _,
        ))
        .with_title("colour-uniform".to_string());
    let backend = create_backend(wb, &event_loop);

    let mut renderer_state = unsafe { RendererState::new(backend) };

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

    renderer_state.draw_triangle(cr, cg, cb);

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
                    winit::event::WindowEvent::RedrawRequested => {
                        renderer_state.draw_triangle(cr, cg, cb);
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
            }
            _ => (),
        };
    });
}

#[cfg(not(any(feature = "vulkan", feature = "metal",)))]
fn main() {
    println!("You need to enable the native API feature (vulkan/metal) in order to test the LL");
}
