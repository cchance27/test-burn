//! Broadcast Element-wise Add Kernel for Foundry.
//!
//! Adds a 1D bias tensor to each row: out[i] = a[i] + b[i % b_len]

use metallic_macros::{Kernel, KernelArgs, MetalStruct, Stage as DeriveStage};

use crate::{
    compound::{CompiledCompoundKernel, CompoundKernel}, spec::DynamicValue, types::TensorArg
};

/// Parameters for ElemwiseAdd kernel.
#[derive(MetalStruct, Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[repr(C)]
pub struct ElemwiseAddParams {
    /// Total elements in output.
    pub total_elements: DynamicValue<u32>,
    /// Length of bias tensor (for broadcast).
    pub b_len: DynamicValue<u32>,
}

/// Broadcast element-wise add kernel.
///
/// out[i] = a[i] + b[i % b_len]
#[derive(Kernel, KernelArgs, Clone, Default)]
#[kernel(
    source = "elemwise/add.metal",
    function = "broadcast_add_kernel_f16",
    args = ElemwiseAddParamsResolved,
    dispatch = per_element,
    dtype = F16,
    step = true
)]
pub struct ElemwiseAdd {
    /// Input tensor a.
    pub a: TensorArg,
    /// Bias tensor b (1D).
    pub b: TensorArg,
    /// Output tensor.
    #[arg(output)]
    pub out: TensorArg,
    /// Kernel parameters.
    pub params: ElemwiseAddParamsResolved,
}

impl ElemwiseAdd {
    /// Create a new broadcast add kernel.
    pub fn new(a: &TensorArg, b: &TensorArg, out: &TensorArg, params: ElemwiseAddParamsResolved) -> Self {
        Self {
            a: a.clone(),
            b: b.clone(),
            out: out.clone(),
            params,
        }
    }

    /// Create inplace variant (out = a).
    pub fn new_inplace(a: &TensorArg, b: &TensorArg, total_elements: u32, b_len: u32) -> Self {
        Self {
            a: a.clone(),
            b: b.clone(),
            out: a.clone(), // Same as input for inplace
            params: ElemwiseAddParamsResolved { total_elements, b_len },
        }
    }

    /// Create a new stage for use in compound kernels.
    pub fn stage() -> ElemwiseAddStage {
        ElemwiseAddStage::default()
    }

    /// Get a compiled kernel for residual addition or standalone broadcast add.
    pub fn get_kernel() -> std::sync::Arc<CompiledCompoundKernel> {
        use crate::kernel_registry::{KernelCacheKey, kernel_registry};
        let key = KernelCacheKey::new("elemwise", "add_broadcast");
        kernel_registry().get_or_build(key, || {
            CompoundKernel::new("elemwise_add_broadcast")
                .main(ElemwiseAdd::stage())
                .with_manual_output(true)
                .compile()
        })
    }
}

#[derive(Clone, Debug, Default, DeriveStage)]
#[stage(
    emit = r#"
    const uint idx = gid.x * tptg.x + lid.x;
    if (idx >= total_elements) return;
    out[idx] = a[idx] + b[idx % b_len];
    "#,
    out_var = "void"
)]
pub struct ElemwiseAddStage {
    #[arg(buffer = 0)]
    pub a: TensorArg,
    #[arg(buffer = 1)]
    pub b: TensorArg,
    #[arg(buffer = 2, output)]
    pub out: TensorArg,
    #[arg(buffer = 3, metal_type = "constant uint&")]
    pub total_elements: u32,
    #[arg(buffer = 4, metal_type = "constant uint&")]
    pub b_len: u32,
}

/// Arguments for dispatching the unified ElemwiseAdd kernel.
#[derive(Debug, KernelArgs)]
#[allow(dead_code)]
pub struct ElemwiseAddArgs {
    #[arg(buffer = 0)]
    pub a: TensorArg,
    #[arg(buffer = 1)]
    pub b: TensorArg,
    #[arg(buffer = 2)]
    pub out: TensorArg,
    #[arg(buffer = 3)]
    pub total_elements: u32,
    #[arg(buffer = 4)]
    pub b_len: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elemwise_add_params_metal_struct() {
        let def = ElemwiseAddParams::METAL_STRUCT_DEF;
        assert!(def.contains("struct ElemwiseAddParams"));
        assert!(def.contains("total_elements"));
    }
}
