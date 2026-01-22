//! Layout Stage - Handles memory layout and indexing for GEMV and other kernels.
//!
//! Memory layout determines how weight tensors are indexed:
//! - `RowMajor`: Weights stored as [N, K] - output dimension first
//! - `ColMajor`: Weights stored as [K, N] - input dimension first
//! - `Canonical`: Blocked [N, K] for cache efficiency

use serde::{Deserialize, Serialize};

use crate::compound::{BufferArg, Stage};

/// Memory layout for 2D weight tensors in GEMV.
///
/// In GEMV, we compute: output[n] = sum_k(input[k] * weights[n, k])
/// The `row` variable in indexing refers to the OUTPUT dimension (N).
///
/// **IMPORTANT**: The naming indicates how weights are stored in memory:
/// - `RowMajor`: Weights are [N, K] shape - each output row is contiguous in K
/// - `ColMajor`: Weights are [K, N] shape - each input column is contiguous in N  
/// - `Canonical`: Blocked [N, K] with K blocked into chunks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Layout {
    /// Weights stored as [N, K] (output-major).
    /// Index: weights[n * K + k]
    /// Each output neuron's weights are contiguous in memory.
    /// Use when weights tensor has shape [N, K].
    RowMajor,

    /// Weights stored as [K, N] (input-major).
    /// Index: weights[k * N + n]
    /// Each input feature's connections are contiguous in memory.
    /// Use when weights tensor has shape [K, N].
    ColMajor,

    /// Canonical Blocked format: [N, K] with K dimension blocked.
    /// Index: weights[(k % wpb) + wpb * (n + (k / wpb) * N)]
    /// Used by legacy GemvCanonical kernels for cache efficiency.
    Canonical,
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

    /// Weights stored as [N, K] - each output row is contiguous in K.
    /// Use for Context transpose_b=true (Context expects [N, K]).
    pub fn row_major() -> Self {
        Self::new(Layout::RowMajor, "gid", "lid")
    }

    /// Weights stored as [K, N] - each input column is contiguous in N.
    /// Use for Context transpose_b=false (Context expects [K, N]).
    pub fn col_major() -> Self {
        Self::new(Layout::ColMajor, "gid", "lid")
    }

    /// Canonical blocked [N, K] format for cache efficiency.
    pub fn canonical() -> Self {
        Self::new(Layout::Canonical, "gid", "lid")
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
// Layout: RowMajor - Weights shape [N, K]
// weights[n * K + k] - output dimension is the slow axis
// Context equivalent: transpose_b=true with [N, K] tensor
#define WEIGHT_STRIDE_K 1
#define WEIGHT_STRIDE_ROW K
#define IS_K_CONTIGUOUS 1
#define WEIGHT_INDEX(row, k, K, N) ((row) * (K) + (k))
"#
            .to_string(),
            Layout::ColMajor => r#"
// Layout: ColMajor - Weights shape [K, N]
// weights[k * N + n] - input dimension is the slow axis
// Context equivalent: transpose_b=false with [K, N] tensor
#define WEIGHT_STRIDE_K N
#define WEIGHT_STRIDE_ROW 1
#define IS_K_CONTIGUOUS 0
#define WEIGHT_INDEX(row, k, K, N) ((k) * (N) + (row))
"#
            .to_string(),
            Layout::Canonical => r#"
// Layout: Canonical Blocked - Weights shape [N, K] with K blocked
// weights[(k % wpb) + wpb * (n + (k / wpb) * N)]
// Used for cache-efficient GEMV with large K
#define IS_K_CONTIGUOUS 1
#define WEIGHT_INDEX(row, k, K, N) ((k % weights_per_block) + weights_per_block * ((row) + ((k) / weights_per_block) * (N)))
"#
            .to_string(),
        }
    }

    fn emit(&self, _prev: &str) -> (String, String) {
        let code = format!(
            r#"    // Layout indices
    uint row_idx = {gid}.x;  // Output dimension index (N)
    uint tid = {lid}.x;
    uint col_idx = tid;
    
    // Layout: {layout_name}
"#,
            gid = self.gid_var,
            lid = self.lid_var,
            layout_name = match self.layout {
                Layout::RowMajor => "[N, K] (output-major)",
                Layout::ColMajor => "[K, N] (input-major)",
                Layout::Canonical => "[N, K] Blocked",
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
    /// Elements processed per thread per iteration (default 8)
    elems_per_thread: u32,
}

impl WarpLayoutStage {
    pub fn new(layout: Layout) -> Self {
        Self {
            layout,
            warps_per_tg: 1,
            simd_width: 32,
            elems_per_thread: 8,
        }
    }

    /// Weights stored as [N, K] - warp-per-row dispatch.
    pub fn row_major() -> Self {
        Self::new(Layout::RowMajor)
    }

    /// Weights stored as [K, N] - warp-per-row dispatch.
    pub fn col_major() -> Self {
        Self::new(Layout::ColMajor)
    }

    /// Canonical blocked [N, K] format - warp-per-row dispatch.
    pub fn canonical() -> Self {
        Self::new(Layout::Canonical)
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

    /// Configure elements per thread.
    pub fn with_elems_per_thread(mut self, elems: u32) -> Self {
        self.elems_per_thread = elems;
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
// Layout: RowMajor - Weights shape [N, K]
// weights[n * K + k] - output dimension is the slow axis
#define WEIGHT_STRIDE_K 1
#define WEIGHT_STRIDE_ROW K
#define IS_K_CONTIGUOUS 1
#define IS_CANONICAL 0
#define WEIGHT_INDEX(row, k, K, N) ((row) * (K) + (k))
"#
            }
            Layout::ColMajor => {
                r#"
// Layout: ColMajor - Weights shape [K, N]
// weights[k * N + n] - input dimension is the slow axis
#define WEIGHT_STRIDE_K N
#define WEIGHT_STRIDE_ROW 1
#define IS_K_CONTIGUOUS 0
#define IS_CANONICAL 0
#define WEIGHT_INDEX(row, k, K, N) ((k) * (N) + (row))
"#
            }
            Layout::Canonical => {
                r#"
// Layout: Canonical Blocked - Weights shape [N, K] with K blocked
// weights[(k % wpb) + wpb * (n + (k / wpb) * N)]
#define IS_K_CONTIGUOUS 1
#define IS_CANONICAL 1
#define WEIGHT_INDEX(row, k, K, N) ((k % weights_per_block) + weights_per_block * ((row) + ((k) / weights_per_block) * (N)))
"#
            }
        };

        format!(
            "{}\n#define WARPS_PER_TG {}\n#define SIMD_WIDTH {}\n#define ELEMS_PER_THREAD {}\n#define K_CHUNK_SIZE (SIMD_WIDTH * ELEMS_PER_THREAD)\n",
            layout_defs, self.warps_per_tg, self.simd_width, self.elems_per_thread
        )
    }

    fn emit(&self, _prev: &str) -> (String, String) {
        let code = format!(
            r#"    // Warp-per-row layout indices
    const uint warp_id = lid.x / SIMD_WIDTH;
    const uint lane_id = lid.x & (SIMD_WIDTH - 1u);
    const uint row_idx = gid.x * WARPS_PER_TG + warp_id;  // Output index (N)
    const uint tid = lane_id;
    const uint batch_idx = gid.y; // Support batched dispatch
    
    // Early exit if row is out of bounds
    if (row_idx >= n_dim) return;
    
    // Layout: {layout_name}
"#,
            layout_name = match self.layout {
                Layout::RowMajor => "[N, K] (output-major)",
                Layout::ColMajor => "[K, N] (input-major)",
                Layout::Canonical => "[N, K] Blocked",
            }
        );

        ("void".to_string(), code)
    }
}

// =============================================================================
// ThreadLayoutStage - Thread-per-row dispatch strategy (Scalar)
// =============================================================================

/// A stage that defines thread-per-row layout for scalar GEMV dispatch.
///
/// Assigns one output row per thread.
/// Efficient for Layout::ColMajor (KxN weights) where contiguous load is along N.
///
/// Dispatch: N threads (Grid = (N+255)/256)
///
/// Emits:
/// - `row_idx`: Row index = gid.x
/// - `lane_id`: 0 (dummy)
#[derive(Debug, Clone)]
pub struct ThreadLayoutStage {
    layout: Layout,
}

impl ThreadLayoutStage {
    pub fn new(layout: Layout) -> Self {
        Self { layout }
    }

    pub fn col_major() -> Self {
        Self::new(Layout::ColMajor)
    }
}

impl Stage for ThreadLayoutStage {
    fn includes(&self) -> Vec<&'static str> {
        vec![]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![]
    }

    fn struct_defs(&self) -> String {
        // Reuse layout definitions
        let layout_defs = match self.layout {
            Layout::ColMajor => {
                r#"
// Layout: ColMajor - Weights shape [K, N]
// Thread-per-row: Contiguous load along N possible if we group?
// Actually if we assume Thread-per-Row, we load W[k, n] where n varies by thread.
// Lane i loads W[k, n+i]. This IS contiguous in memory (stride 1 along N).
#define WEIGHT_STRIDE_K N
#define WEIGHT_STRIDE_ROW 1
#define IS_K_CONTIGUOUS 0
#define WEIGHT_INDEX(row, k, K, N) ((k) * (N) + (row))
"#
            }
            _ => panic!("ThreadLayoutStage currently only supports ColMajor (Scalar) optimization"),
        };

        format!("{}\n#define SIMD_WIDTH 32\n", layout_defs)
    }

    fn emit(&self, _prev: &str) -> (String, String) {
        let code = r#"    // Thread-per-row layout indices
	    const uint row_idx = gid.x;
	    const uint lane_id = 0; // Scalar mode implies single lane acting as 'warp'
	    const uint tid = lid.x;
	    const uint batch_idx = gid.y; // Support batched dispatch (gid.y==0 for d1 grids)

	    if (row_idx >= n_dim) return;
	    
	    // Define scalar constants
	    #define SCALAR_DISPATCH 1
"#
        .to_string();

        ("void".to_string(), code)
    }
}
