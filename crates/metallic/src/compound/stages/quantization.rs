//! Quantization types for staged kernels.
//!
//! Centralizes quantization handling so adding a new quant type only requires:
//! 1. New Metal policy file (e.g., `policies/policy_q4.metal`)
//! 2. New enum variant here

use serde::{Deserialize, Serialize};

/// Quantization types for weight loading.
///
/// Each variant maps to a `PolicyXXX` struct in Metal that implements
/// `load_weights<N>()` and `load_scale()` methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Quantization {
    /// FP16 weights (no quantization, scale = 1.0)
    F16,
    /// 8-bit quantization (Q8_0 or similar)
    Q8,
    // Future: Q4, INT8, etc.
}

impl Quantization {
    /// Metal include file path for this quantization policy.
    pub fn include_path(&self) -> &'static str {
        match self {
            Quantization::F16 => "policies/policy_f16.metal",
            Quantization::Q8 => "policies/policy_q8.metal",
        }
    }

    /// Metal policy struct name.
    pub fn policy_name(&self) -> &'static str {
        match self {
            Quantization::F16 => "PolicyF16",
            Quantization::Q8 => "PolicyQ8",
        }
    }

    /// Short name for kernel naming.
    pub fn short_name(&self) -> &'static str {
        match self {
            Quantization::F16 => "f16",
            Quantization::Q8 => "q8",
        }
    }
}

impl Default for Quantization {
    fn default() -> Self {
        Quantization::F16
    }
}
