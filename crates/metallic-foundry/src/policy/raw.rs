use metallic_loader::LoadedModel;
use metallic_macros::MetalPolicy;
use objc2_metal::MTLResourceOptions;

use super::{LoaderStage, MetalPolicyRuntime};
use crate::{Foundry, compound::Layout, types::TensorArg};

#[derive(Debug, MetalPolicy, Clone, Default)]
#[policy(
    header = "policies/policy_u32.metal",
    struct_name = "PolicyU32",
    short_name = "u32",
    element_size = 4,
    block_size = 1,
    vector_load_size = 1,
    unroll_factor = 1,
    active_thread_count = 1,
    has_scale = false
)]
pub struct PolicyU32;

impl MetalPolicyRuntime for PolicyU32 {
    fn loader_stage(&self) -> Box<dyn LoaderStage> {
        // Fallback to F16 loader if no specific U32 loader exists.
        // This is mostly for kernel includes/registration.
        Box::new(super::f16::PolicyF16)
    }

    fn load_weights(
        &self,
        foundry: &mut Foundry,
        model: &dyn LoadedModel,
        source_tensor_name: &str,
        logical_name: &str,
        _layout: Layout,
    ) -> anyhow::Result<Vec<(String, TensorArg)>> {
        let tensor_info = model
            .tensor_info(source_tensor_name)
            .ok_or_else(|| anyhow::anyhow!("Tensor {} not found", source_tensor_name))?;
        let dims = &tensor_info.dimensions;
        let data = model.tensor_data(source_tensor_name)?;
        let data_bytes = data.as_slice();

        let buffer = foundry
            .device
            .new_buffer_from_slice(data_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow::anyhow!("Failed to allocate buffer for {}", source_tensor_name))?;

        let tensor_arg = TensorArg::from_buffer(buffer, crate::tensor::Dtype::U32, dims.clone(), vec![1; dims.len()]);

        Ok(vec![(logical_name.to_string(), tensor_arg)])
    }
}
