extern crate gfx_hal as hal;

use hal::image::Extent;
use hal::window::Extent2D;
use winit::dpi::{LogicalSize, PhysicalSize};

pub struct ScreenSizeState {
    dpi_factor: f64,
    size: LogicalSize<u32>,
}

// TODO: Could write tests for this if I wanted to
impl ScreenSizeState {
    pub fn new(width: u32, height: u32, dpi_factor: f64) -> Self {
        ScreenSizeState {
            dpi_factor,
            size: LogicalSize::new(width, height),
        }
    }

    pub fn new_default_min(dpi_factor: f64) -> Self {
        Self::new(64, 64, dpi_factor)
    }

    pub fn new_default_start(dpi_factor: f64) -> Self {
        Self::new(1024, 768, dpi_factor)
    }

    pub fn physical_size(&self) -> PhysicalSize<u32> {
        self.size.to_physical(self.dpi_factor)
    }

    pub fn logical_size(&self) -> LogicalSize<u32> {
        self.size
    }

    pub fn extent_2d(&self) -> Extent2D {
        let p_size = self.physical_size();
        Extent2D {
            width: p_size.width as _,
            height: p_size.height as _,
        }
    }

    pub fn extent(&self) -> Extent {
        let p_size = self.physical_size();
        // TODO: Figure out why depth is generally set to 1
        Extent {
            width: p_size.width as _,
            height: p_size.height as _,
            depth: 1,
        }
    }

    pub fn set_dpi_factor(&mut self, dpi_factor: f64) {
        self.dpi_factor = dpi_factor;
    }

    pub fn set_size(&mut self, size: LogicalSize<u32>) {
        self.size = size;
    }

    pub fn set_physical_size(&mut self, p_size: PhysicalSize<u32>) {
        self.size = LogicalSize::from_physical(p_size, self.dpi_factor);
    }
}
