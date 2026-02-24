use std::{str::FromStr, sync::Arc};

use anyhow::Result;

use super::MetalPolicyRuntime;
use crate::tensor::Dtype;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DenseStorageVariant {
    F16,
    #[default]
    Preserve,
}

impl DenseStorageVariant {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            DenseStorageVariant::F16 => "f16",
            DenseStorageVariant::Preserve => "preserve",
        }
    }
}

impl FromStr for DenseStorageVariant {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f16" => Ok(Self::F16),
            "preserve" | "native" | "preserve_dense" => Ok(Self::Preserve),
            other => Err(anyhow::anyhow!("unsupported dense storage variant '{other}'")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantComputeVariant {
    #[default]
    F16,
    BF16,
    F32,
}

impl QuantComputeVariant {
    #[must_use]
    pub const fn dtype(self) -> Dtype {
        match self {
            QuantComputeVariant::F16 => Dtype::F16,
            QuantComputeVariant::BF16 => Dtype::BF16,
            QuantComputeVariant::F32 => Dtype::F32,
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            QuantComputeVariant::F16 => "f16",
            QuantComputeVariant::BF16 => "bf16",
            QuantComputeVariant::F32 => "f32",
        }
    }
}

impl FromStr for QuantComputeVariant {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f16" => Ok(Self::F16),
            "bf16" => Ok(Self::BF16),
            "f32" => Ok(Self::F32),
            other => Err(anyhow::anyhow!("unsupported quant compute variant '{other}'")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PolicyVariant {
    pub dense_storage: DenseStorageVariant,
    pub quant_compute: QuantComputeVariant,
}

impl PolicyVariant {
    #[must_use]
    pub const fn preserve_dense(quant_compute: QuantComputeVariant) -> Self {
        Self {
            dense_storage: DenseStorageVariant::Preserve,
            quant_compute,
        }
    }
}

impl FromStr for PolicyVariant {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Ok(Self::preserve_dense(QuantComputeVariant::F16));
        }

        if let Ok(dense) = DenseStorageVariant::from_str(trimmed) {
            return Ok(Self {
                dense_storage: dense,
                quant_compute: QuantComputeVariant::F16,
            });
        }

        let lowered = trimmed.to_ascii_lowercase();
        if let Some(raw_dense) = lowered.strip_prefix("dense:") {
            let dense_storage = DenseStorageVariant::from_str(raw_dense)?;
            return Ok(Self {
                dense_storage,
                quant_compute: QuantComputeVariant::F16,
            });
        }
        if let Some(raw_quant) = lowered.strip_prefix("preserve:") {
            let quant_compute = QuantComputeVariant::from_str(raw_quant)?;
            return Ok(Self::preserve_dense(quant_compute));
        }

        let mut dense_storage = None;
        let mut quant_compute = None;
        for raw_part in lowered.split([',', ';']) {
            let part = raw_part.trim();
            if part.is_empty() {
                continue;
            }
            let (key, raw_val) = part
                .split_once('=')
                .ok_or_else(|| anyhow::anyhow!("invalid policy variant segment '{part}' (expected key=value)"))?;
            let key = key.trim();
            let raw_val = raw_val.trim();
            match key {
                "dense" | "dense_storage" => dense_storage = Some(DenseStorageVariant::from_str(raw_val)?),
                "quant" | "quant_compute" | "compute" => quant_compute = Some(QuantComputeVariant::from_str(raw_val)?),
                other => return Err(anyhow::anyhow!("unsupported policy variant key '{other}'")),
            }
        }

        if dense_storage.is_none() && quant_compute.is_none() {
            return Err(anyhow::anyhow!(
                "unsupported policy variant '{trimmed}' (expected preserve/preserve:<dtype> or key=value list)"
            ));
        }

        Ok(Self {
            dense_storage: dense_storage.unwrap_or(DenseStorageVariant::Preserve),
            quant_compute: quant_compute.unwrap_or(QuantComputeVariant::F16),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolicyResolution {
    pub source_dtype: Dtype,
    pub storage_dtype: Dtype,
    pub compute_dtype: Dtype,
    pub lossy_cast: bool,
}

#[derive(Debug, Clone)]
pub struct ResolvedPolicy {
    pub policy: Arc<dyn MetalPolicyRuntime>,
    pub resolution: PolicyResolution,
}

pub fn resolve_policy_for_dtype(dtype: Dtype, variant: PolicyVariant) -> Result<ResolvedPolicy> {
    let quant_compute = variant.quant_compute.dtype();

    let resolved = match dtype {
        Dtype::F16 => ResolvedPolicy {
            policy: Arc::new(super::f16::PolicyF16),
            resolution: PolicyResolution {
                source_dtype: Dtype::F16,
                storage_dtype: Dtype::F16,
                compute_dtype: Dtype::F16,
                lossy_cast: false,
            },
        },
        Dtype::F32 => ResolvedPolicy {
            policy: match variant.dense_storage {
                DenseStorageVariant::F16 => Arc::new(super::f32::PolicyF32),
                DenseStorageVariant::Preserve => Arc::new(super::f32_native::PolicyF32Native),
            },
            resolution: match variant.dense_storage {
                DenseStorageVariant::F16 => PolicyResolution {
                    source_dtype: Dtype::F32,
                    storage_dtype: Dtype::F16,
                    compute_dtype: Dtype::F16,
                    lossy_cast: true,
                },
                DenseStorageVariant::Preserve => PolicyResolution {
                    source_dtype: Dtype::F32,
                    storage_dtype: Dtype::F32,
                    compute_dtype: Dtype::F32,
                    lossy_cast: false,
                },
            },
        },
        Dtype::Q4_0 => quant_policy(Arc::new(super::q4_0::PolicyQ4_0), dtype, quant_compute),
        Dtype::Q4_1 => quant_policy(Arc::new(super::q4_1::PolicyQ4_1), dtype, quant_compute),
        Dtype::Q5_K => quant_policy(Arc::new(super::q5_k::PolicyQ5K), dtype, quant_compute),
        Dtype::Q6_K => quant_policy(Arc::new(super::q6_k::PolicyQ6K), dtype, quant_compute),
        Dtype::Q8_0 => quant_policy(Arc::new(super::q8::PolicyQ8), dtype, quant_compute),
        Dtype::U32 => ResolvedPolicy {
            policy: Arc::new(super::raw::PolicyU32),
            resolution: PolicyResolution {
                source_dtype: Dtype::U32,
                storage_dtype: Dtype::U32,
                compute_dtype: Dtype::U32,
                lossy_cast: false,
            },
        },
        other => {
            return Err(anyhow::anyhow!(
                "Unsupported tensor dtype {:?} for Foundry (add a policy or convert the model).",
                other
            )
            .context(format!(
                "policy variant dense={} quant_compute={}",
                variant.dense_storage.as_str(),
                variant.quant_compute.as_str()
            )));
        }
    };

    Ok(resolved)
}

fn quant_policy(policy: Arc<dyn MetalPolicyRuntime>, source_dtype: Dtype, quant_compute: Dtype) -> ResolvedPolicy {
    ResolvedPolicy {
        policy,
        resolution: PolicyResolution {
            source_dtype,
            storage_dtype: source_dtype,
            compute_dtype: quant_compute,
            lossy_cast: false,
        },
    }
}

#[path = "variant.test.rs"]
mod tests;
