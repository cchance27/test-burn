use super::{LoaderStage, OptimizationMetadata, QuantizationPolicy, WeightLayout};
use crate::{
    Foundry, compound::{BufferArg, stages::Quantization}, dtypes::F16, gguf::{file::GGUFDataType, model_loader::GGUFModel, tensor_info::GGUFRawTensor}, spec::{FastBindings, ResolvedSymbols}, tensor::{Tensor, TensorInit}, types::TensorArg
};

#[derive(Debug, Clone, Default)]
pub struct PolicyF16;

impl crate::compound::Stage for PolicyF16 {
    fn includes(&self) -> Vec<&'static str> {
        vec![]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![]
    }

    fn emit(&self, _input_var: &str) -> (String, String) {
        // Loader stages don't emit code directly; they are used by the Policy template.
        ("".to_string(), "".to_string())
    }
}

impl LoaderStage for PolicyF16 {
    fn params_struct(&self) -> String {
        // F16 doesn't need extra params in the loader struct
        "".to_string()
    }

    fn bind(&self, fast_bindings: &FastBindings, resolved: &ResolvedSymbols) -> smallvec::SmallVec<[TensorArg; 4]> {
        use smallvec::smallvec;
        let tensor = fast_bindings.get(resolved.weights).expect("F16 weight bound");

        // Match existing behavior: [weight, dummy_scale]
        // where dummy_scale is just the weight again (unused by kernel)
        // DEBT: We should avoid this, or at least assess is as it seems like a hack and might affect memory?
        smallvec![tensor.clone(), tensor.clone()]
    }
    fn quantization_type(&self) -> Quantization {
        Quantization::F16
    }
}

impl QuantizationPolicy for PolicyF16 {
    fn name(&self) -> &'static str {
        "F16"
    }

    fn metal_policy_name(&self) -> &'static str {
        "PolicyF16"
    }

    fn metal_include(&self) -> &'static str {
        "policies/policy_f16.metal"
    }

    fn optimization_hints(&self) -> OptimizationMetadata {
        OptimizationMetadata {
            block_size: 1,
            vector_load_size: 2,
            unroll_factor: 4,
            active_thread_count: 32,
        }
    }

    fn loader_stage(&self) -> Box<dyn LoaderStage> {
        Box::new(self.clone())
    }

    fn load_weights(
        &self,
        foundry: &mut Foundry,
        gguf: &GGUFModel,
        gguf_tensor_name: &str,
        logical_name: &str,
        layout: WeightLayout,
    ) -> anyhow::Result<Vec<(String, TensorArg)>> {
        let tensor_info = gguf
            .get_tensor(gguf_tensor_name)
            .ok_or_else(|| anyhow::anyhow!("Tensor {} not found in GGUF", gguf_tensor_name))?;

        let dims: Vec<usize> = tensor_info.dims().to_vec();

        // Read raw as F16
        let view = tensor_info.raw_view(&gguf.gguf_file)?;
        let data_f16: Vec<half::f16> = match view {
            GGUFRawTensor::F16(v) => v.to_vec(),
            GGUFRawTensor::Bytes(bytes, GGUFDataType::F16) => bytemuck::try_cast_slice(bytes)
                .map_err(|e| anyhow::anyhow!("Invalid F16 bytes: {}", e))?
                .to_vec(),
            _ => {
                return Err(anyhow::anyhow!(
                    "PolicyF16 expects F16 GGUF data. Got {:?}",
                    tensor_info.data_type()
                ));
            }
        };

        if let WeightLayout::Canonical { expected_k, expected_n } = layout {
            const WEIGHTS_PER_BLOCK: usize = 32;

            if !((dims[0] == expected_k && dims[1] == expected_n) || (dims[0] == expected_n && dims[1] == expected_k)) {
                return Err(anyhow::anyhow!(
                    "Canonical dims mismatch for '{}': got {:?}, exp ({}, {})",
                    gguf_tensor_name,
                    dims,
                    expected_k,
                    expected_n
                ));
            }

            let blocks_per_k = expected_k.div_ceil(WEIGHTS_PER_BLOCK);
            let canonical_len = blocks_per_k * expected_n * WEIGHTS_PER_BLOCK;
            let mut canonical_data = vec![half::f16::from_f32(0.0); canonical_len];

            let layout_hint = gguf.layout_hint();

            for out_idx in 0..expected_n {
                for block in 0..blocks_per_k {
                    let k_base = block * WEIGHTS_PER_BLOCK;
                    let dst_base = (block * expected_n + out_idx) * WEIGHTS_PER_BLOCK;
                    let remaining = expected_k.saturating_sub(k_base);

                    if remaining >= WEIGHTS_PER_BLOCK {
                        for i in 0..WEIGHTS_PER_BLOCK {
                            let k_idx = k_base + i;
                            let src_idx = match layout_hint {
                                crate::gguf::model_loader::GGUFLayoutHint::Nk => out_idx * expected_k + k_idx,
                                crate::gguf::model_loader::GGUFLayoutHint::Kn => k_idx * expected_n + out_idx,
                            };
                            canonical_data[dst_base + i] = data_f16[src_idx];
                        }
                    } else if remaining > 0 {
                        for i in 0..remaining {
                            let k_idx = k_base + i;
                            let src_idx = match layout_hint {
                                crate::gguf::model_loader::GGUFLayoutHint::Nk => out_idx * expected_k + k_idx,
                                crate::gguf::model_loader::GGUFLayoutHint::Kn => k_idx * expected_n + out_idx,
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
            let layout_hint = gguf.layout_hint();

            if dims.len() == 2 {
                let (n, k) = match layout_hint {
                    crate::gguf::model_loader::GGUFLayoutHint::Nk => (dims[1], dims[0]),
                    crate::gguf::model_loader::GGUFLayoutHint::Kn => (dims[0], dims[1]),
                };

                let data_to_upload = if matches!(layout_hint, crate::gguf::model_loader::GGUFLayoutHint::Kn) {
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
