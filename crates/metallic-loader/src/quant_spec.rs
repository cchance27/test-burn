use metallic_sdk::tensor::Dtype;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantBlockSpec {
    pub weights_per_block: usize,
    pub block_bytes: usize,
}

impl QuantBlockSpec {
    pub const fn new(weights_per_block: usize, block_bytes: usize) -> Self {
        Self {
            weights_per_block,
            block_bytes,
        }
    }
}

pub const GGML_QK_K: usize = 256;

pub const Q4_0_SPEC: QuantBlockSpec = QuantBlockSpec::new(32, 2 + 16);
pub const Q4_1_SPEC: QuantBlockSpec = QuantBlockSpec::new(32, 2 + 2 + 16);
pub const Q5_0_SPEC: QuantBlockSpec = QuantBlockSpec::new(32, 2 + 4 + 16);
pub const Q5_1_SPEC: QuantBlockSpec = QuantBlockSpec::new(32, 2 + 2 + 4 + 16);
pub const Q8_0_SPEC: QuantBlockSpec = QuantBlockSpec::new(32, 2 + 32);
pub const Q8_1_SPEC: QuantBlockSpec = QuantBlockSpec::new(32, 4 + 4 + 32);
pub const Q2_K_SPEC: QuantBlockSpec = QuantBlockSpec::new(GGML_QK_K, 2 + 2 + GGML_QK_K / 16 + GGML_QK_K / 4);
pub const Q3_K_SPEC: QuantBlockSpec = QuantBlockSpec::new(GGML_QK_K, 2 + GGML_QK_K / 4 + GGML_QK_K / 8 + 12);
pub const Q4_K_SPEC: QuantBlockSpec = QuantBlockSpec::new(GGML_QK_K, 2 + 2 + GGML_QK_K / 2 + 12);
pub const Q5_K_SPEC: QuantBlockSpec = QuantBlockSpec::new(GGML_QK_K, 2 + 2 + GGML_QK_K / 2 + GGML_QK_K / 8 + 12);
pub const Q6_K_SPEC: QuantBlockSpec = QuantBlockSpec::new(GGML_QK_K, 2 + GGML_QK_K / 2 + GGML_QK_K / 4 + GGML_QK_K / 16);
pub const Q8_K_SPEC: QuantBlockSpec = QuantBlockSpec::new(GGML_QK_K, 4 + GGML_QK_K + GGML_QK_K / 8);
pub const TQ1_0_SPEC: QuantBlockSpec = QuantBlockSpec::new(GGML_QK_K, 2 + 4 * 13);
pub const TQ2_0_SPEC: QuantBlockSpec = QuantBlockSpec::new(GGML_QK_K, 2 + GGML_QK_K / 4);
pub const MXFP4_SPEC: QuantBlockSpec = QuantBlockSpec::new(32, 1 + 16);

pub fn block_quant_spec_for_dtype(dtype: Dtype) -> Option<QuantBlockSpec> {
    match dtype {
        Dtype::Q4_0 => Some(Q4_0_SPEC),
        Dtype::Q4_1 => Some(Q4_1_SPEC),
        Dtype::Q5_0 => Some(Q5_0_SPEC),
        Dtype::Q5_1 => Some(Q5_1_SPEC),
        Dtype::Q8_0 => Some(Q8_0_SPEC),
        Dtype::Q8_1 => Some(Q8_1_SPEC),
        Dtype::Q2_K => Some(Q2_K_SPEC),
        Dtype::Q3_K => Some(Q3_K_SPEC),
        Dtype::Q4_K => Some(Q4_K_SPEC),
        Dtype::Q5_K => Some(Q5_K_SPEC),
        Dtype::Q6_K => Some(Q6_K_SPEC),
        Dtype::Q8_K => Some(Q8_K_SPEC),
        _ => None,
    }
}

pub fn quantized_tensor_storage_bytes_for_dtype(dtype: Dtype, dims: &[usize]) -> Option<usize> {
    let spec = block_quant_spec_for_dtype(dtype)?;
    let element_count = dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))?;
    let blocks = element_count.div_ceil(spec.weights_per_block);
    blocks.checked_mul(spec.block_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q6_k_spec_matches_ggml_block_constants() {
        assert_eq!(Q6_K_SPEC.weights_per_block, 256);
        assert_eq!(Q6_K_SPEC.block_bytes, 210);
    }

    #[test]
    fn q6_k_storage_size_matches_known_tensor_shape() {
        let size = quantized_tensor_storage_bytes_for_dtype(Dtype::Q6_K, &[4864, 896]).expect("q6_k size");
        assert_eq!(size, 3_575_040);
    }
}
