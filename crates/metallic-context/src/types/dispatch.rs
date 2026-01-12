//! Dispatch configuration types for Metal compute kernels.
//!
//! This module provides pure-Rust types for dispatch configuration,
//! hiding `objc2_metal::MTLSize` from consumers.

use objc2_metal::MTLSize;

/// Size of the compute grid (number of threadgroups).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GridSize {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

impl GridSize {
    pub const fn new(width: usize, height: usize, depth: usize) -> Self {
        Self { width, height, depth }
    }

    /// Create a 1D grid.
    pub const fn d1(width: usize) -> Self {
        Self {
            width,
            height: 1,
            depth: 1,
        }
    }

    /// Create a 2D grid.
    pub const fn d2(width: usize, height: usize) -> Self {
        Self { width, height, depth: 1 }
    }
}

/// Size of a threadgroup (threads per threadgroup).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ThreadgroupSize {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

impl ThreadgroupSize {
    pub const fn new(width: usize, height: usize, depth: usize) -> Self {
        Self { width, height, depth }
    }

    /// Create a 1D threadgroup.
    pub const fn d1(width: usize) -> Self {
        Self {
            width,
            height: 1,
            depth: 1,
        }
    }

    /// Create a 2D threadgroup.
    pub const fn d2(width: usize, height: usize) -> Self {
        Self { width, height, depth: 1 }
    }
}

/// Dispatch configuration for a compute kernel.
///
/// Encapsulates both the grid size (number of threadgroups) and
/// threadgroup size (threads per threadgroup).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DispatchConfig {
    pub grid: GridSize,
    pub group: ThreadgroupSize,
}

impl DispatchConfig {
    pub const fn new(grid: GridSize, group: ThreadgroupSize) -> Self {
        Self { grid, group }
    }

    /// Create a 1D dispatch configuration.
    pub const fn d1(grid_width: usize, group_width: usize) -> Self {
        Self {
            grid: GridSize::d1(grid_width),
            group: ThreadgroupSize::d1(group_width),
        }
    }
}

// --- Internal conversions to MTLSize (not exposed to consumers) ---

impl From<GridSize> for MTLSize {
    fn from(g: GridSize) -> Self {
        MTLSize {
            width: g.width,
            height: g.height,
            depth: g.depth,
        }
    }
}

impl From<ThreadgroupSize> for MTLSize {
    fn from(t: ThreadgroupSize) -> Self {
        MTLSize {
            width: t.width,
            height: t.height,
            depth: t.depth,
        }
    }
}

impl From<DispatchConfig> for (MTLSize, MTLSize) {
    fn from(config: DispatchConfig) -> Self {
        (config.grid.into(), config.group.into())
    }
}

// --- Conversions from MTLSize (for migration/interop) ---

impl From<MTLSize> for GridSize {
    fn from(s: MTLSize) -> Self {
        Self {
            width: s.width,
            height: s.height,
            depth: s.depth,
        }
    }
}

impl From<MTLSize> for ThreadgroupSize {
    fn from(s: MTLSize) -> Self {
        Self {
            width: s.width,
            height: s.height,
            depth: s.depth,
        }
    }
}

impl From<(MTLSize, MTLSize)> for DispatchConfig {
    fn from((grid, group): (MTLSize, MTLSize)) -> Self {
        Self {
            grid: grid.into(),
            group: group.into(),
        }
    }
}
