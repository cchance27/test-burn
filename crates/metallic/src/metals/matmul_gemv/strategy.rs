/// Quantization Policy (The "What")
///
/// Defines the data type and loading behavior.
/// Corresponds to a Metal struct that provides `load()` methods.
pub trait QuantPolicy: Send + Sync + 'static {
    /// Metal struct name for this quantization policy.
    /// e.g. "Q8", "F16"
    fn metal_name() -> &'static str;

    /// C-struct definition for runtime parameters (pointers, etc.)
    fn struct_def() -> &'static str;
}

/// Execution Strategy (The "How")
///
/// Defines the loop structure, threading model, and unrolling.
/// e.g. "Canonical"
pub trait GemvStrategy: Send + Sync + 'static {
    /// The base template name in Metal.
    /// e.g. "simd_gemv_canonical"
    fn template_name() -> &'static str;
}

use crate::tensor::Dtype;

/// Auto-detection trait for dispatch
pub trait AutoQuant: QuantPolicy {
    /// Checks if this policy matches the runtime DType.
    fn valid_for_dtype(dtype: Dtype) -> bool;
}

/// F16 Quantization Policy
#[derive(Clone, Copy, Debug, Default)]
pub struct F16;

impl QuantPolicy for F16 {
    fn metal_name() -> &'static str {
        "F16"
    }

    fn struct_def() -> &'static str {
        r#"
        struct F16Params {
            const device half **data;
            const device half *gamma;
            float inv_rms;
            uint weights_per_block;
        };
        "#
    }
}

impl AutoQuant for F16 {
    fn valid_for_dtype(dtype: Dtype) -> bool {
        matches!(dtype, Dtype::F16)
    }
}

/// Q8 Quantization Policy
#[derive(Clone, Copy, Debug, Default)]
pub struct Q8;

impl QuantPolicy for Q8 {
    fn metal_name() -> &'static str {
        "Q8"
    }

    fn struct_def() -> &'static str {
        r#"
        struct Q8Params {
            const device uchar **data;
            const device uchar **scale_bytes;
            const device half *gamma;
            float inv_rms;
            uint weights_per_block;
        };
        "#
    }
}

impl AutoQuant for Q8 {
    fn valid_for_dtype(dtype: Dtype) -> bool {
        matches!(dtype, Dtype::U8)
    }
}

use std::marker::PhantomData;

/// Canonical GEMV Strategy
///
/// Standard decode implementation (1 token).
/// Unrolls 4x, manages barrier synchronization.
#[derive(Clone, Copy, Debug, Default)]
pub struct Canonical<Q: QuantPolicy> {
    _q: PhantomData<Q>,
}

impl<Q: QuantPolicy> GemvStrategy for Canonical<Q> {
    fn template_name() -> &'static str {
        "simd_gemv_canonical"
    }
}
