use std::{fmt::Debug, sync::Arc};

use anyhow::Result;
use metallic_loader::LoadedModel;

use crate::{
    compound::Layout, spec::{FastBindings, ResolvedSymbols}, types::TensorArg
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

/// Compatibility alias for the unified Layout enum.
pub type WeightLayout = Layout;

use crate::compound::Stage;

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
    fn quantization_type(&self) -> Arc<dyn MetalPolicyRuntime>;
}

/// Runtime behavior for a Metal policy (loading weights, binding buffers).
///
/// This trait extends `MetalPolicy` which provides compile-time kernel generation metadata.
/// `MetalPolicyRuntime` adds the runtime loading and binding capabilities.
pub trait MetalPolicyRuntime: crate::fusion::MetalPolicy + Send + Sync + Debug {
    /// Return the stage responsible for loading weights/scales.
    /// This stage defines the necessary Metal buffer arguments.
    fn loader_stage(&self) -> Box<dyn LoaderStage>;

    /// Load weights from a loaded model, potentially splitting or transforming them.
    ///
    /// Returns a list of (logical_name, TensorArg) pairs to insert into the Foundry workspace.
    /// - `logical_name`: The canonical name the model expects (e.g. "blk.0.attn.q_proj.weight").
    ///   Note: Policies like Q8 might return multiple pairs, e.g. "name" (weights) and "name_scales".
    fn load_weights(
        &self,
        foundry: &mut crate::Foundry,
        model: &dyn LoadedModel,
        source_tensor_name: &str,
        logical_name: &str,
        layout: Layout,
    ) -> Result<Vec<(String, TensorArg)>>;
}

pub mod activation;
pub(crate) mod block_quant;
pub mod f16;
pub mod f32;
pub mod f64;
pub mod q4_0;
pub mod q8;
pub mod raw;

use crate::tensor::Dtype;

/// Registry to retrieve policies by name or stringified dtype.
// DEBT: This is very fragile! we should clean this up having fragile string matches?
pub fn resolve_policy(dtype: Dtype) -> Arc<dyn MetalPolicyRuntime> {
    match dtype {
        Dtype::F16 => Arc::new(f16::PolicyF16),
        Dtype::F32 => Arc::new(f32::PolicyF32),
        Dtype::Q4_0 => Arc::new(q4_0::PolicyQ4_0),
        Dtype::Q8_0 => Arc::new(q8::PolicyQ8),
        Dtype::U32 => Arc::new(raw::PolicyU32),
        // Map everything else to F32 or panic?
        // Legacy string match panicked on unsupported.
        // Assuming F64 -> F32 as before?
        _ => panic!(
            "Unsupported tensor dtype {:?} for Foundry (add a policy or convert the model).",
            dtype
        ),
    }
}

/// Resolve a policy by its short name (e.g. "f16", "q8").
// DEBT: This is very fragile! we should clean this up having fragile string matches?
pub fn resolve_policy_by_name(name: &str) -> Option<Arc<dyn MetalPolicyRuntime>> {
    match name {
        "f16" => Some(Arc::new(f16::PolicyF16)),
        "f32" => Some(Arc::new(f32::PolicyF32)),
        "f64" => Some(Arc::new(f64::PolicyF64)),
        "u32" => Some(Arc::new(raw::PolicyU32)),
        "q4_0" => Some(Arc::new(q4_0::PolicyQ4_0)),
        "q8" | "q8_0" => Some(Arc::new(q8::PolicyQ8)),
        _ => None,
    }
}

impl PartialEq for dyn MetalPolicyRuntime + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.short_name() == other.short_name()
    }
}
impl Eq for dyn MetalPolicyRuntime + '_ {}
impl std::hash::Hash for dyn MetalPolicyRuntime + '_ {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.short_name().hash(state);
    }
}

pub mod serde {
    use ::serde::{Deserialize, Deserializer, Serializer};

    use super::*;

    pub fn serialize<S>(policy: &Arc<dyn MetalPolicyRuntime>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(policy.short_name())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Arc<dyn MetalPolicyRuntime>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        resolve_policy_by_name(&s).ok_or_else(|| ::serde::de::Error::custom(format!("Unknown policy: {}", s)))
    }
}

pub mod serde_option {
    use ::serde::{Deserialize, Deserializer, Serializer};

    use super::*;

    pub fn serialize<S>(policy: &Option<Arc<dyn MetalPolicyRuntime>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match policy {
            Some(p) => serializer.serialize_str(p.short_name()),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Arc<dyn MetalPolicyRuntime>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s: Option<String> = Option::deserialize(deserializer)?;
        match s {
            Some(name) => {
                let p = resolve_policy_by_name(&name).ok_or_else(|| ::serde::de::Error::custom(format!("Unknown policy: {}", name)))?;
                Ok(Some(p))
            }
            None => Ok(None),
        }
    }
}
