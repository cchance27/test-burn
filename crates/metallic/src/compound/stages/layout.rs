//! Layout Stage - Handles memory layout and indexing for GEMV and other kernels.
//!
//! Supports:
//! - RowMajor (NK): weights[row * K + k] - rows contiguous
//! - ColMajor (KN): weights[k * N + row] - columns contiguous

use serde::{Deserialize, Serialize};

use crate::compound::{BufferArg, Stage};

/// Memory layout for 2D tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Layout {
    /// Row-major (NK): weights[row * K + k]
    /// Each row is contiguous in memory.
    RowMajor,
    /// Column-major (KN): weights[k * N + row]  
    /// Each column is contiguous in memory.
    ColMajor,
}

/// A stage that defines layout indexing variables and helpers.
///
/// Emits:
/// - `row_idx`: Row index from threadgroup position
/// - `tid`: Thread ID within threadgroup
/// - `col_idx`: Column index (alias for tid)
/// - Layout-specific stride and indexing macros
#[derive(Debug, Clone)]
pub struct LayoutStage {
    layout: Layout,
    gid_var: String,
    lid_var: String,
}

impl LayoutStage {
    pub fn new(layout: Layout, gid_var: impl Into<String>, lid_var: impl Into<String>) -> Self {
        Self {
            layout,
            gid_var: gid_var.into(),
            lid_var: lid_var.into(),
        }
    }

    /// Row-major (NK) layout: weights[row * K + k]
    pub fn row_major() -> Self {
        Self::new(Layout::RowMajor, "gid", "lid")
    }

    /// Column-major (KN) layout: weights[k * N + row]
    pub fn col_major() -> Self {
        Self::new(Layout::ColMajor, "gid", "lid")
    }

    /// Get the layout type
    pub fn layout(&self) -> Layout {
        self.layout
    }
}

impl Stage for LayoutStage {
    fn includes(&self) -> Vec<&'static str> {
        vec![]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![]
    }

    fn struct_defs(&self) -> String {
        // Emit layout-specific indexing macros
        match self.layout {
            Layout::RowMajor => r#"
// Layout: Row-Major (NK)
// weights[row * K + k] - rows are contiguous
#define WEIGHT_STRIDE_K 1
#define WEIGHT_STRIDE_ROW K
#define IS_K_CONTIGUOUS 1
#define WEIGHT_INDEX(row, k, K, N) ((row) * (K) + (k))
"#
            .to_string(),
            Layout::ColMajor => r#"
// Layout: Column-Major (KN)  
// weights[k * N + row] - columns are contiguous
#define WEIGHT_STRIDE_K N
#define WEIGHT_STRIDE_ROW 1
#define IS_K_CONTIGUOUS 0
#define WEIGHT_INDEX(row, k, K, N) ((k) * (N) + (row))
"#
            .to_string(),
        }
    }

    fn emit(&self, _prev: &str) -> (String, String) {
        let code = format!(
            r#"    // Layout indices
    uint row_idx = {gid}.x;
    uint tid = {lid}.x;
    uint col_idx = tid;
    
    // Layout type: {layout_name}
"#,
            gid = self.gid_var,
            lid = self.lid_var,
            layout_name = match self.layout {
                Layout::RowMajor => "RowMajor (NK)",
                Layout::ColMajor => "ColMajor (KN)",
            }
        );

        ("void".to_string(), code)
    }
}

// =============================================================================
// WarpLayoutStage - Warp-per-row dispatch strategy
// =============================================================================

/// A stage that defines warp-per-row layout for optimized GEMV dispatch.
///
/// Unlike `LayoutStage` which assigns one row per threadgroup, this assigns
/// one row per warp (simd_width threads), allowing multiple rows per TG.
///
/// Dispatch: (N + warps_per_tg - 1) / warps_per_tg threadgroups × TG_WIDTH threads
///
/// Emits:
/// - `warp_id`: Warp index within threadgroup
/// - `lane_id`: Thread index within warp (0..simd_width-1)
/// - `row_idx`: Row index computed from gid.x * warps_per_tg + warp_id
/// - Layout-specific WEIGHT_INDEX macro
#[derive(Debug, Clone)]
pub struct WarpLayoutStage {
    layout: Layout,
    /// Number of warps per threadgroup (typically 8 for 256 threads)
    warps_per_tg: u32,
    /// SIMD width (32 for Apple Silicon)
    simd_width: u32,
}

impl WarpLayoutStage {
    pub fn new(layout: Layout) -> Self {
        Self {
            layout,
            warps_per_tg: 8,
            simd_width: 32,
        }
    }

    /// Row-major (NK) layout with warp-per-row dispatch.
    pub fn row_major() -> Self {
        Self::new(Layout::RowMajor)
    }

    /// Column-major (KN) layout with warp-per-row dispatch.
    pub fn col_major() -> Self {
        Self::new(Layout::ColMajor)
    }

    /// Configure number of warps per threadgroup.
    pub fn with_warps(mut self, warps: u32) -> Self {
        self.warps_per_tg = warps;
        self
    }

    /// Configure SIMD width.
    pub fn with_simd_width(mut self, width: u32) -> Self {
        self.simd_width = width;
        self
    }

    /// Total threads per threadgroup.
    pub fn threads_per_tg(&self) -> usize {
        (self.warps_per_tg * self.simd_width) as usize
    }

    /// Get the layout type.
    pub fn layout(&self) -> Layout {
        self.layout
    }
}

impl Stage for WarpLayoutStage {
    fn includes(&self) -> Vec<&'static str> {
        vec![]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![]
    }

    fn struct_defs(&self) -> String {
        // Emit layout-specific indexing macros + warp constants
        let layout_defs = match self.layout {
            Layout::RowMajor => {
                r#"
// Layout: Row-Major (NK)
// weights[row * K + k] - rows are contiguous
#define WEIGHT_STRIDE_K 1
#define WEIGHT_STRIDE_ROW K
#define IS_K_CONTIGUOUS 1
#define WEIGHT_INDEX(row, k, K, N) ((row) * (K) + (k))
"#
            }
            Layout::ColMajor => {
                r#"
// Layout: Column-Major (KN)  
// weights[k * N + row] - columns are contiguous
#define WEIGHT_STRIDE_K N
#define WEIGHT_STRIDE_ROW 1
#define IS_K_CONTIGUOUS 0
#define WEIGHT_INDEX(row, k, K, N) ((k) * (N) + (row))
"#
            }
        };

        format!(
            "{}\n#define WARPS_PER_TG {}\n#define SIMD_WIDTH {}\n#define ELEMS_PER_THREAD 8\n#define K_CHUNK_SIZE (SIMD_WIDTH * ELEMS_PER_THREAD)\n",
            layout_defs, self.warps_per_tg, self.simd_width
        )
    }

    fn emit(&self, _prev: &str) -> (String, String) {
        let code = format!(
            r#"    // Warp-per-row layout indices
    const uint warp_id = lid.x / SIMD_WIDTH;
    const uint lane_id = lid.x & (SIMD_WIDTH - 1u);
    const uint row_idx = gid.x * WARPS_PER_TG + warp_id;
    const uint tid = lane_id;
    
    // Early exit if row is out of bounds
    if (row_idx >= n_dim) return;
    
    // Layout type: {layout_name}
"#,
            layout_name = match self.layout {
                Layout::RowMajor => "RowMajor (NK) with warp-per-row",
                Layout::ColMajor => "ColMajor (KN) with warp-per-row",
            }
        );

        ("void".to_string(), code)
    }
}
