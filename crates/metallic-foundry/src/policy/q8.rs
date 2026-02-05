use metallic_loader::LoadedModel;
use metallic_macros::MetalPolicy;

use super::{LoaderStage, MetalPolicyRuntime};
use crate::{
    Foundry, compound::Layout, spec::{FastBindings, ResolvedSymbols}, tensor::Dtype, types::TensorArg
};

const Q8_0_WPB: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34;
const Q8_0_SCALE_BYTES: usize = 2;
const Q8_0_DATA_BYTES: usize = 32;

#[inline]
fn q8_0_write_block(qs: &[u8], out: &mut [u8]) {
    debug_assert_eq!(qs.len(), Q8_0_DATA_BYTES);
    debug_assert_eq!(out.len(), Q8_0_DATA_BYTES);
    out.copy_from_slice(qs);
}

#[derive(Debug, Clone, Default, MetalPolicy)]
#[policy(
    header = "policies/policy_q8.metal",
    struct_name = "PolicyQ8",
    short_name = "q8",
    element_size = 1,
    block_size = 32,
    vector_load_size = 8,
    unroll_factor = 2,
    active_thread_count = 32,
    block_size_bytes = 34,
    weights_per_block = 32
)]
pub struct PolicyQ8;

// PolicyQ8 implements MetalPolicy via derive macro.
// Custom validation logic if needed should be added to the macro or a separate trait if we change it again,
// but for now we follow the project pattern of using derives for boilerplate.

impl LoaderStage for PolicyQ8 {
    fn params_struct(&self) -> String {
        "".to_string()
    }

    fn bind(&self, fast_bindings: &FastBindings, resolved: &ResolvedSymbols) -> smallvec::SmallVec<[TensorArg; 4]> {
        use smallvec::smallvec;
        // Q8 expects [weight, scales]
        // We panic here if bindings are missing because bind assumes indices are valid
        // and presence was validated during compile/load (or we accept the panic as "unbound tensor").
        // For performance, we can unwrap or use expect.
        let weight = fast_bindings.get(resolved.weights).expect("Q8 weight bound");

        let scales_idx = resolved.scales.expect("Q8 requires scales index");
        let scales = fast_bindings.get(scales_idx).expect("Q8 scales bound");

        smallvec![weight.clone(), scales.clone()]
    }

    fn quantization_type(&self) -> std::sync::Arc<dyn MetalPolicyRuntime> {
        std::sync::Arc::new(PolicyQ8)
    }
}

impl MetalPolicyRuntime for PolicyQ8 {
    fn loader_stage(&self) -> Box<dyn LoaderStage> {
        Box::new(self.clone())
    }

    fn load_weights(
        &self,
        foundry: &mut Foundry,
        model: &dyn LoadedModel,
        source_tensor_name: &str,
        logical_name: &str,
        layout: Layout,
    ) -> anyhow::Result<Vec<(String, TensorArg)>> {
        let loaded = crate::policy::block_quant::load_block_quant_2d::<Q8_0_WPB, Q8_0_BLOCK_BYTES, Q8_0_SCALE_BYTES, Q8_0_DATA_BYTES>(
            foundry,
            model,
            source_tensor_name,
            Dtype::Q8_0,
            Dtype::Q8_0,
            layout,
            q8_0_write_block,
        )?;

        Ok(vec![
            (logical_name.to_string(), loaded.weights),
            (format!("{}_scales", logical_name), loaded.scales),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_canonical_reorders_scales_and_data_consistently() {
        let target_n = 3;
        let blocks_per_k = 2;
        let total_blocks = target_n * blocks_per_k;

        let mut raw = vec![0u8; total_blocks * Q8_0_BLOCK_BYTES];
        for src_block_idx in 0..total_blocks {
            let start = src_block_idx * Q8_0_BLOCK_BYTES;
            raw[start] = src_block_idx as u8; // scale byte 0
            raw[start + 1] = 0xEE; // scale byte 1 sentinel
            raw[start + Q8_0_SCALE_BYTES..start + Q8_0_BLOCK_BYTES].fill((0x10 + src_block_idx) as u8);
        }

        let mut data_out = vec![0u8; total_blocks * Q8_0_WPB];
        let mut scales_out = vec![0u8; total_blocks * Q8_0_SCALE_BYTES];

        crate::policy::block_quant::split_blocks::<Q8_0_BLOCK_BYTES, Q8_0_SCALE_BYTES, Q8_0_DATA_BYTES>(
            &raw,
            blocks_per_k,
            target_n,
            true,
            &mut data_out,
            &mut scales_out,
            q8_0_write_block,
        );

        for src_block_idx in 0..total_blocks {
            let scale_start = src_block_idx * Q8_0_SCALE_BYTES;
            assert_eq!(scales_out[scale_start], src_block_idx as u8);
            assert_eq!(scales_out[scale_start + 1], 0xEE);

            let dst = crate::policy::block_quant::canonical_dst_block_idx(src_block_idx, blocks_per_k, target_n);
            let data_start = dst * Q8_0_WPB;
            assert_eq!(data_out[data_start], (0x10 + src_block_idx) as u8);
            assert_eq!(data_out[data_start + Q8_0_WPB - 1], (0x10 + src_block_idx) as u8);
        }
    }
}
