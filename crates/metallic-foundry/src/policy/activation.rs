use serde::{Deserialize, Serialize};

/// Supported activation functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Activation {
    #[default]
    #[serde(alias = "none", alias = "ActivationNone", alias = "")]
    None,
    #[serde(alias = "silu", alias = "ActivationSiLU")]
    SiLU,
    #[serde(alias = "relu", alias = "ActivationReLU")]
    ReLU,
    #[serde(alias = "gelu", alias = "ActivationGELU")]
    GELU,
}

impl Activation {
    /// Metal header to include for all activations.
    #[inline]
    pub const fn header(self) -> &'static str {
        "policies/activations.metal"
    }

    /// Metal struct name implementing this activation.
    #[inline]
    pub const fn struct_name(self) -> &'static str {
        match self {
            Activation::None => "ActivationNone",
            Activation::SiLU => "ActivationSiLU",
            Activation::ReLU => "ActivationReLU",
            Activation::GELU => "ActivationGELU",
        }
    }

    /// Optimized lookup for common names (for GGUF compatibility).
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "silu" | "ActivationSiLU" => Some(Activation::SiLU),
            "relu" | "ActivationReLU" => Some(Activation::ReLU),
            "gelu" | "ActivationGELU" => Some(Activation::GELU),
            "none" | "ActivationNone" | "" => Some(Activation::None),
            _ => None,
        }
    }
}
