use std::ptr;

use objc2_metal::{MTLBuffer as _, MTLDevice};

use super::{LoaderStage, OptimizationMetadata, QuantizationPolicy, WeightLayout};
use crate::{
    compound::{BufferArg, stages::Quantization}, foundry::{
        Foundry, spec::{FastBindings, SymbolTable}
    }, gguf::{file::GGUFDataType, model_loader::GGUFModel, tensor_info::GGUFRawTensor}, tensor::Dtype, types::{MetalBuffer, TensorArg}
};

#[derive(Debug, Clone)]
pub struct Q8LoaderStage;

impl crate::compound::Stage for Q8LoaderStage {
    fn includes(&self) -> Vec<&'static str> {
        vec![]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![]
    }

    fn emit(&self, _input_var: &str) -> (String, String) {
        ("".to_string(), "".to_string())
    }
}

impl LoaderStage for Q8LoaderStage {
    fn params_struct(&self) -> String {
        "".to_string()
    }

    fn bind(&self, fast_bindings: &FastBindings, base_name: &str, symbol_table: &SymbolTable) -> anyhow::Result<Vec<TensorArg>> {
        let weight_sym = symbol_table
            .get(base_name)
            .ok_or_else(|| anyhow::anyhow!("Symbol '{}' not found", base_name))?;
        let weight = fast_bindings
            .get(weight_sym)
            .ok_or_else(|| anyhow::anyhow!("Q8 binding '{}' not found", base_name))?;

        let scales_name = format!("{}_scales", base_name);
        let scales_sym = symbol_table
            .get(&scales_name)
            .ok_or_else(|| anyhow::anyhow!("Symbol '{}' not found", scales_name))?;

        let scales = fast_bindings
            .get(scales_sym)
            .ok_or_else(|| anyhow::anyhow!("Q8 binding '{}' not found", scales_name))?;

        Ok(vec![weight.clone(), scales.clone()])
    }

    fn quantization_type(&self) -> Quantization {
        Quantization::Q8
    }
}

#[derive(Debug)]
pub struct PolicyQ8;

impl QuantizationPolicy for PolicyQ8 {
    fn name(&self) -> &'static str {
        "Q8_0"
    }

    fn metal_policy_name(&self) -> &'static str {
        "PolicyQ8"
    }

    fn metal_include(&self) -> &'static str {
        "policies/policy_q8.metal"
    }

    fn optimization_hints(&self) -> OptimizationMetadata {
        OptimizationMetadata {
            block_size: 32,      // Q8 works on blocks of 32
            vector_load_size: 8, // can load 8 bytes (8 ints) at once often
            unroll_factor: 2,
            active_thread_count: 32,
        }
    }

    fn loader_stage(&self) -> Box<dyn LoaderStage> {
        Box::new(Q8LoaderStage)
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
        if dims.len() != 2 {
            return Err(anyhow::anyhow!("Q8_0 tensor '{}' must be 2D (got {:?})", gguf_tensor_name, dims));
        }

        const WPB: usize = 32;
        const BLOCK_BYTES: usize = 34; // 2 bytes scale + 32 bytes data
        const SCALE_BYTES: usize = 2;

        let (target_k, target_n, is_canonical) = match layout {
            WeightLayout::RowMajor => (dims[1], dims[0], false),
            WeightLayout::Canonical { expected_k, expected_n } => (expected_k, expected_n, true),
        };

        let view = tensor_info.raw_view(&gguf.gguf_file)?;
        let raw = match view {
            GGUFRawTensor::Bytes(b, GGUFDataType::Q8_0) => b,
            _ => {
                return Err(anyhow::anyhow!(
                    "Tensor '{}' is not Q8_0 bytes. Got {:?}",
                    gguf_tensor_name,
                    tensor_info.data_type()
                ));
            }
        };

        let contiguous_dim_len = if is_canonical {
            if !((dims[0] == target_k && dims[1] == target_n) || (dims[0] == target_n && dims[1] == target_k)) {
                return Err(anyhow::anyhow!(
                    "Q8 canonical tensor '{}' dims {:?} mismatch (K,N)=({},{})",
                    gguf_tensor_name,
                    dims,
                    target_k,
                    target_n
                ));
            }
            target_k
        } else {
            dims[1]
        };

        if contiguous_dim_len % WPB != 0 {
            return Err(anyhow::anyhow!(
                "Q8_0 tensor '{}' contig dim {} not divisible by 32",
                gguf_tensor_name,
                contiguous_dim_len
            ));
        }

        // Block count for contiguous dim
        let blocks_per_k = contiguous_dim_len / WPB;
        let n_dim_len = if is_canonical { target_n } else { dims[0] };

        let total_blocks = n_dim_len * blocks_per_k;
        let expected_bytes = total_blocks * BLOCK_BYTES;

        if raw.len() != expected_bytes {
            return Err(anyhow::anyhow!(
                "Q8 tensor size mismatch for '{}': got {}, exp {}",
                gguf_tensor_name,
                raw.len(),
                expected_bytes
            ));
        }

        // Allocate separate Metal buffers
        let data_len = total_blocks * WPB;
        let scales_len = total_blocks * SCALE_BYTES;

        let data_buffer = foundry
            .device
            .0
            .newBufferWithLength_options(data_len, objc2_metal::MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow::anyhow!("Failed to allocate Q8 data"))?;

        let scales_buffer = foundry
            .device
            .0
            .newBufferWithLength_options(scales_len, objc2_metal::MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow::anyhow!("Failed to allocate Q8 scales"))?;

        if is_canonical {
            if gguf.layout_hint() != crate::gguf::model_loader::GGUFLayoutHint::Nk {
                return Err(anyhow::anyhow!("Q8 canonical requires Nk layout"));
            }

            unsafe {
                let data_ptr = data_buffer.contents().as_ptr() as *mut u8;
                let scales_ptr = scales_buffer.contents().as_ptr() as *mut u8;

                for (src_block_idx, chunk) in raw.chunks_exact(BLOCK_BYTES).enumerate() {
                    let row = src_block_idx / blocks_per_k;
                    let block = src_block_idx - row * blocks_per_k;

                    ptr::copy_nonoverlapping(chunk.as_ptr(), scales_ptr.add(src_block_idx * SCALE_BYTES), SCALE_BYTES);

                    let dst_block_idx = block * target_n + row;
                    ptr::copy_nonoverlapping(chunk.as_ptr().add(SCALE_BYTES), data_ptr.add(dst_block_idx * WPB), WPB);
                }
            }
        } else {
            // Standard Split
            unsafe {
                let data_ptr = data_buffer.contents().as_ptr() as *mut u8;
                let scales_ptr = scales_buffer.contents().as_ptr() as *mut u8;

                for (i, chunk) in raw.chunks_exact(BLOCK_BYTES).enumerate() {
                    ptr::copy_nonoverlapping(chunk.as_ptr(), scales_ptr.add(i * SCALE_BYTES), SCALE_BYTES);
                    ptr::copy_nonoverlapping(chunk.as_ptr().add(SCALE_BYTES), data_ptr.add(i * WPB), WPB);
                }
            }
        }

        let data_arg = TensorArg::from_buffer(
            MetalBuffer(data_buffer),
            Dtype::U8,
            if is_canonical { vec![data_len] } else { dims.clone() },
            if is_canonical {
                vec![1]
            } else {
                crate::foundry::tensor::compute_strides(&dims)
            },
        );

        let scales_arg = TensorArg::from_buffer(MetalBuffer(scales_buffer), Dtype::U8, vec![scales_len], vec![1]);

        Ok(vec![
            (logical_name.to_string(), data_arg),
            (format!("{}_scales", logical_name), scales_arg),
        ])
    }
}
