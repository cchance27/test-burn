use super::GGUFDataType;
use crate::quant_spec::{
    GGML_QK_K, MXFP4_SPEC, Q2_K_SPEC, Q3_K_SPEC, Q4_0_SPEC, Q4_1_SPEC, Q4_K_SPEC, Q5_0_SPEC, Q5_1_SPEC, Q5_K_SPEC, Q6_K_SPEC, Q8_0_SPEC, Q8_1_SPEC, Q8_K_SPEC, QuantBlockSpec, TQ1_0_SPEC, TQ2_0_SPEC
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GGUFDtypeClass {
    Scalar { element_size_bytes: usize },
    BlockQuant(QuantBlockSpec),
    Deprecated { replacement_hint: &'static str },
    Unsupported { reason: &'static str },
}

pub fn classify_gguf_dtype(dtype: GGUFDataType) -> GGUFDtypeClass {
    match dtype {
        GGUFDataType::F32 => GGUFDtypeClass::Scalar { element_size_bytes: 4 },
        GGUFDataType::F16 => GGUFDtypeClass::Scalar { element_size_bytes: 2 },
        GGUFDataType::Q4_0 => GGUFDtypeClass::BlockQuant(Q4_0_SPEC),
        GGUFDataType::Q4_1 => GGUFDtypeClass::BlockQuant(Q4_1_SPEC),
        GGUFDataType::Q4_2 => GGUFDtypeClass::Deprecated {
            replacement_hint: "Legacy GGUF type id; add explicit compatibility decode/repack if you need to load it",
        },
        GGUFDataType::Q4_3 => GGUFDtypeClass::Deprecated {
            replacement_hint: "Legacy GGUF type id; add explicit compatibility decode/repack if you need to load it",
        },
        GGUFDataType::Q5_0 => GGUFDtypeClass::BlockQuant(Q5_0_SPEC),
        GGUFDataType::Q5_1 => GGUFDtypeClass::BlockQuant(Q5_1_SPEC),
        GGUFDataType::Q8_0 => GGUFDtypeClass::BlockQuant(Q8_0_SPEC),
        GGUFDataType::Q8_1 => GGUFDtypeClass::BlockQuant(Q8_1_SPEC),
        GGUFDataType::Q2K => GGUFDtypeClass::BlockQuant(Q2_K_SPEC),
        GGUFDataType::Q3K => GGUFDtypeClass::BlockQuant(Q3_K_SPEC),
        GGUFDataType::Q4K => GGUFDtypeClass::BlockQuant(Q4_K_SPEC),
        GGUFDataType::Q5K => GGUFDtypeClass::BlockQuant(Q5_K_SPEC),
        GGUFDataType::Q6K => GGUFDtypeClass::BlockQuant(Q6_K_SPEC),
        GGUFDataType::Q8K => GGUFDtypeClass::BlockQuant(Q8_K_SPEC),
        GGUFDataType::IQ2XXS => GGUFDtypeClass::BlockQuant(QuantBlockSpec::new(GGML_QK_K, 2 + GGML_QK_K / 4)),
        GGUFDataType::IQ2XS => GGUFDtypeClass::BlockQuant(QuantBlockSpec::new(GGML_QK_K, 2 + GGML_QK_K / 4 + GGML_QK_K / 32)),
        GGUFDataType::IQ3XXS => GGUFDtypeClass::BlockQuant(QuantBlockSpec::new(GGML_QK_K, 2 + GGML_QK_K / 4 + GGML_QK_K / 8)),
        GGUFDataType::IQ1S => GGUFDtypeClass::BlockQuant(QuantBlockSpec::new(GGML_QK_K, 2 + GGML_QK_K / 8 + GGML_QK_K / 16)),
        GGUFDataType::IQ4NL => GGUFDtypeClass::BlockQuant(QuantBlockSpec::new(32, 2 + 16)),
        GGUFDataType::IQ3S => GGUFDtypeClass::BlockQuant(QuantBlockSpec::new(
            GGML_QK_K,
            2 + GGML_QK_K / 4 + GGML_QK_K / 8 + GGML_QK_K / 32 + 4,
        )),
        GGUFDataType::IQ2S => GGUFDtypeClass::BlockQuant(QuantBlockSpec::new(GGML_QK_K, 2 + GGML_QK_K / 4 + GGML_QK_K / 16)),
        GGUFDataType::IQ4XS => GGUFDtypeClass::BlockQuant(QuantBlockSpec::new(GGML_QK_K, 2 + 2 + GGML_QK_K / 2 + GGML_QK_K / 64)),
        GGUFDataType::I8 => GGUFDtypeClass::Scalar { element_size_bytes: 1 },
        GGUFDataType::I16 => GGUFDtypeClass::Scalar { element_size_bytes: 2 },
        GGUFDataType::I32 => GGUFDtypeClass::Scalar { element_size_bytes: 4 },
        GGUFDataType::I64 => GGUFDtypeClass::Scalar { element_size_bytes: 8 },
        GGUFDataType::F64 => GGUFDtypeClass::Scalar { element_size_bytes: 8 },
        GGUFDataType::IQ1M => GGUFDtypeClass::BlockQuant(QuantBlockSpec::new(GGML_QK_K, GGML_QK_K / 8 + GGML_QK_K / 16 + GGML_QK_K / 32)),
        GGUFDataType::BF16 => GGUFDtypeClass::Scalar { element_size_bytes: 2 },
        GGUFDataType::Q4044 => GGUFDtypeClass::Deprecated {
            replacement_hint: "Legacy tiled Q4_0 GGUF id; support via explicit runtime repack path if required",
        },
        GGUFDataType::Q4048 => GGUFDtypeClass::Deprecated {
            replacement_hint: "Legacy tiled Q4_0 GGUF id; support via explicit runtime repack path if required",
        },
        GGUFDataType::Q4088 => GGUFDtypeClass::Deprecated {
            replacement_hint: "Legacy tiled Q4_0 GGUF id; support via explicit runtime repack path if required",
        },
        GGUFDataType::TQ10 => GGUFDtypeClass::BlockQuant(TQ1_0_SPEC),
        GGUFDataType::TQ20 => GGUFDtypeClass::BlockQuant(TQ2_0_SPEC),
        GGUFDataType::MXFP4 => GGUFDtypeClass::BlockQuant(MXFP4_SPEC),
        GGUFDataType::IQ4NL44 => GGUFDtypeClass::Deprecated {
            replacement_hint: "Legacy tiled IQ4_NL GGUF id; support via explicit runtime repack path if required",
        },
        GGUFDataType::IQ4NL48 => GGUFDtypeClass::Deprecated {
            replacement_hint: "Legacy tiled IQ4_NL GGUF id; support via explicit runtime repack path if required",
        },
        GGUFDataType::IQ4NL88 => GGUFDtypeClass::Deprecated {
            replacement_hint: "Legacy tiled IQ4_NL GGUF id; support via explicit runtime repack path if required",
        },
        GGUFDataType::Array => GGUFDtypeClass::Unsupported {
            reason: "Array is metadata-only and not a valid tensor storage dtype",
        },
        GGUFDataType::Unknown(_) => GGUFDtypeClass::Unsupported {
            reason: "Unknown GGUF dtype id",
        },
    }
}

pub fn block_quant_spec_for_gguf_dtype(dtype: GGUFDataType) -> Option<QuantBlockSpec> {
    match classify_gguf_dtype(dtype) {
        GGUFDtypeClass::BlockQuant(spec) => Some(spec),
        _ => None,
    }
}

pub fn quantized_tensor_storage_bytes_for_gguf_dtype(dtype: GGUFDataType, dims: &[u64]) -> Option<usize> {
    let spec = block_quant_spec_for_gguf_dtype(dtype)?;
    let element_count = dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(usize::try_from(d).ok()?))?;
    let blocks = element_count.div_ceil(spec.weights_per_block);
    blocks.checked_mul(spec.block_bytes)
}

pub fn scalar_element_size_bytes(dtype: GGUFDataType) -> Option<usize> {
    match classify_gguf_dtype(dtype) {
        GGUFDtypeClass::Scalar { element_size_bytes } => Some(element_size_bytes),
        _ => None,
    }
}

#[path = "quant_spec.test.rs"]
mod tests;
