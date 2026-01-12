use std::{fmt::Debug, sync::Arc};

use anyhow::Result;

use crate::{
    gguf::{file::GGUFDataType, model_loader::GGUFModel}, spec::{FastBindings, ResolvedSymbols}, types::TensorArg
};

/// Optimization hints for kernel dispatch.
#[derive(Debug, Clone, Copy)]
pub struct OptimizationMetadata {
    /// Number of elements processed per thread in the inner loop (e.g., 1 for F16, 32 for Q8 blocks).
    pub block_size: usize,
    /// Vector load size in bytes (e.g., 2 for half, 8 for distinct vectorized loads).
    pub vector_load_size: usize,
    /// Hint for loop unrolling factor.
    pub unroll_factor: usize,
    /// Proposed active thread count per threadgroup (e.g. 32 for SIMD-aligned ops).
    pub active_thread_count: usize,
}

impl Default for OptimizationMetadata {
    fn default() -> Self {
        Self {
            block_size: 1,
            vector_load_size: 2, // half
            unroll_factor: 1,
            active_thread_count: 32,
        }
    }
}

/// Requested layout for the loaded weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightLayout {
    /// Standard row-major layout (NK or KN as per GGUF).
    /// Used for standard GEMM/GEMV.
    RowMajor,
    /// Canonical K-block-major layout for specific kernels (e.g. FusedQkv).
    /// Requires specifying (k, n) dimensions explicitly for valid reordering.
    Canonical { expected_k: usize, expected_n: usize },
}

use crate::compound::{Stage, stages::Quantization};

/// A stage responsible for defining and binding loader arguments.
///
/// This stage defines the Metal struct for loader parameters (e.g., scales, block size)
/// and handles the binding of these arguments from the workspace.
pub trait LoaderStage: Stage {
    /// Return the Metal struct for loader params (e.g. scales, block size).
    fn params_struct(&self) -> String;

    /// Bind arguments using pre-resolved indices (Hot Path).
    ///
    /// This uses integer indices stored in `ResolvedSymbols` to retrieve tensors directly from `FastBindings`.
    fn bind(&self, fast_bindings: &FastBindings, resolved: &ResolvedSymbols) -> smallvec::SmallVec<[TensorArg; 4]>;

    /// Return the quantization type this loader handles.
    fn quantization_type(&self) -> Quantization;
}

/// A policy defining how a specific quantization type is loaded, bound, and executed.
pub trait QuantizationPolicy: Send + Sync + Debug {
    /// Unique identifier for this policy (e.g. "Q8_0", "F16").
    fn name(&self) -> &'static str;

    /// The name of the struct in Metal that implements the Policy interface.
    /// E.g. "PolicyQ8", "PolicyF16".
    fn metal_policy_name(&self) -> &'static str;

    /// The path or content of the Metal header defining the policy.
    fn metal_include(&self) -> &'static str;

    /// Optimization hints for runtime dispatch (threadgroup sizing, etc).
    fn optimization_hints(&self) -> OptimizationMetadata;

    /// Return the stage responsible for loading weights/scales.
    /// This stage defines the necessary Metal buffer arguments.
    fn loader_stage(&self) -> Box<dyn LoaderStage>;

    /// Load weights from GGUF, potentially splitting or transforming them.
    ///
    /// Returns a list of (logical_name, TensorArg) pairs to insert into the Foundry workspace.
    /// - `logical_name`: The canonical name the model expects (e.g. "blk.0.attn.q_proj.weight").
    ///   Note: Policies like Q8 might return multiple pairs, e.g. "name" (weights) and "name_scales".
    fn load_weights(
        &self,
        foundry: &mut crate::Foundry,
        gguf: &GGUFModel,
        gguf_tensor_name: &str,
        logical_name: &str,
        layout: WeightLayout,
    ) -> Result<Vec<(String, TensorArg)>>;
}

pub mod f16;
pub mod f32;
pub mod q8;

use crate::tensor::Dtype;

impl From<Dtype> for GGUFDataType {
    fn from(dtype: Dtype) -> Self {
        match dtype {
            Dtype::F16 => GGUFDataType::F16,
            Dtype::F32 => GGUFDataType::F32,
            Dtype::U8 => GGUFDataType::Q8_0,
            _ => GGUFDataType::Unknown(0),
        }
    }
}

/// Registry to retrieve policies by GGUF type or name.
pub fn resolve_policy(dtype: GGUFDataType) -> Arc<dyn QuantizationPolicy> {
    match dtype {
        GGUFDataType::Q8_0 => Arc::new(q8::PolicyQ8),
        GGUFDataType::F32 => Arc::new(f32::PolicyF32),
        // Default to F16 for F16 and anything else we blindly treat as F16 for now (legacy behavior)
        _ => Arc::new(f16::PolicyF16),
    }
}
