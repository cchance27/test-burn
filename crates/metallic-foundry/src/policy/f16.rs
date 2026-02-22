use metallic_loader::LoadedModel;
use metallic_macros::MetalPolicy;

use super::{LoaderStage, MetalPolicyRuntime, WeightLayout};
use crate::{
    F16, Foundry, spec::{FastBindings, ResolvedSymbols}, tensor::{Tensor, TensorInit}, types::TensorArg
};

#[derive(Debug, Clone, Default, MetalPolicy)]
#[policy(
    header = "policies/policy_f16.metal",
    struct_name = "PolicyF16",
    short_name = "f16",
    element_size = 2,
    block_size = 1,
    vector_load_size = 4,
    unroll_factor = 4,
    active_thread_count = 32,
    has_scale = false
)]
pub struct PolicyF16;

impl LoaderStage for PolicyF16 {
    fn params_struct(&self) -> String {
        // F16 doesn't need extra params in the loader struct
        "".to_string()
    }

    fn bind(&self, fast_bindings: &FastBindings, resolved: &ResolvedSymbols) -> smallvec::SmallVec<[TensorArg; 4]> {
        use smallvec::smallvec;
        let tensor = fast_bindings.get(resolved.weights).expect("F16 weight bound");
        smallvec![tensor.clone()]
    }
    fn quantization_type(&self) -> std::sync::Arc<dyn super::MetalPolicyRuntime> {
        std::sync::Arc::new(PolicyF16)
    }
}

impl MetalPolicyRuntime for PolicyF16 {
    fn loader_stage(&self) -> Box<dyn LoaderStage> {
        Box::new(self.clone())
    }

    fn load_weights(
        &self,
        foundry: &mut Foundry,
        model: &dyn LoadedModel,
        source_tensor_name: &str,
        logical_name: &str,
        layout: WeightLayout,
    ) -> anyhow::Result<Vec<(String, TensorArg)>> {
        let tensor_info = model
            .tensor_info(source_tensor_name)
            .ok_or_else(|| anyhow::anyhow!("Tensor {} not found in model", source_tensor_name))?;

        let dims: Vec<usize> = tensor_info.dimensions.clone();
        let tensor_data = model.tensor_data(source_tensor_name)?;
        let data_bytes = tensor_data.as_slice();

        // Read raw as F16
        let data_f16: Vec<half::f16> = bytemuck::try_cast_slice(data_bytes)
            .map_err(|e| anyhow::anyhow!("Invalid F16 bytes for {}: {}", source_tensor_name, e))?
            .to_vec();

        let layout_hint = model
            .metadata()
            .get_string("metallic.gguf.layout_hint")
            .unwrap_or(std::borrow::Cow::Borrowed("nk"));
        let is_kn = layout_hint == "kn";

        if let WeightLayout::Canonical { expected_k, expected_n } = layout {
            const WEIGHTS_PER_BLOCK: usize = 32;

            if !((dims[0] == expected_k && dims[1] == expected_n) || (dims[0] == expected_n && dims[1] == expected_k)) {
                return Err(anyhow::anyhow!(
                    "Canonical dims mismatch for '{}': got {:?}, exp ({}, {})",
                    source_tensor_name,
                    dims,
                    expected_k,
                    expected_n
                ));
            }

            let blocks_per_k = expected_k.div_ceil(WEIGHTS_PER_BLOCK);
            let canonical_len = blocks_per_k * expected_n * WEIGHTS_PER_BLOCK;
            let mut canonical_data = vec![half::f16::from_f32(0.0); canonical_len];

            for out_idx in 0..expected_n {
                for block in 0..blocks_per_k {
                    let k_base = block * WEIGHTS_PER_BLOCK;
                    let dst_base = (block * expected_n + out_idx) * WEIGHTS_PER_BLOCK;
                    let remaining = expected_k.saturating_sub(k_base);

                    if remaining >= WEIGHTS_PER_BLOCK {
                        for i in 0..WEIGHTS_PER_BLOCK {
                            let k_idx = k_base + i;
                            let src_idx = if !is_kn {
                                out_idx * expected_k + k_idx
                            } else {
                                k_idx * expected_n + out_idx
                            };
                            canonical_data[dst_base + i] = data_f16[src_idx];
                        }
                    } else if remaining > 0 {
                        for i in 0..remaining {
                            let k_idx = k_base + i;
                            let src_idx = if !is_kn {
                                out_idx * expected_k + k_idx
                            } else {
                                k_idx * expected_n + out_idx
                            };
                            canonical_data[dst_base + i] = data_f16[src_idx];
                        }
                    }
                }
            }

            let tensor = Tensor::<F16>::new(foundry, vec![canonical_len], TensorInit::CopyFrom(&canonical_data[..]))
                .map_err(|e| anyhow::anyhow!("Metal error: {:?}", e))?;
            Ok(vec![(logical_name.to_string(), TensorArg::from_tensor(&tensor))])
        } else {
            // RowMajor: ensure data is in [N, K] format for 2D, or preserve 1D
            if dims.len() == 2 {
                let (n, k) = if !is_kn { (dims[0], dims[1]) } else { (dims[1], dims[0]) };

                let data_to_upload = if is_kn {
                    // Transpose Kn [K, n] -> Nk [n, k]
                    let mut transposed = vec![half::f16::from_f32(0.0); n * k];
                    for row_k in 0..k {
                        for col_n in 0..n {
                            transposed[col_n * k + row_k] = data_f16[row_k * n + col_n];
                        }
                    }
                    transposed
                } else {
                    data_f16
                };

                let tensor = Tensor::<F16>::new(foundry, vec![n, k], TensorInit::CopyFrom(&data_to_upload[..]))
                    .map_err(|e| anyhow::anyhow!("Metal error: {:?}", e))?;
                Ok(vec![(logical_name.to_string(), TensorArg::from_tensor(&tensor))])
            } else {
                // 1D or higher dimensionality: just upload as-is
                let tensor = Tensor::<F16>::new(foundry, dims.clone(), TensorInit::CopyFrom(&data_f16[..]))
                    .map_err(|e| anyhow::anyhow!("Metal error: {:?}", e))?;
                Ok(vec![(logical_name.to_string(), TensorArg::from_tensor(&tensor))])
            }
        }
    }
}
