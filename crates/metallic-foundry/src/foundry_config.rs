use std::collections::HashMap;

use metallic_env::{ACCUM_DTYPE, COMPUTE_DTYPE};

use crate::tensor::Dtype;

/// Runtime configuration for Foundry initialization.
///
/// Values here override process environment lookups through `metallic_env` for
/// the lifetime of the created `Foundry`.
#[derive(Clone, Debug, Default)]
pub struct FoundryConfig {
    env_overrides: HashMap<String, String>,
}

impl FoundryConfig {
    /// Set/override a runtime key for this Foundry instance.
    #[must_use]
    pub fn with_env_override(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env_overrides.insert(key.into(), value.into());
        self
    }

    /// Set `METALLIC_ACCUM_DTYPE` using a strongly-typed dtype.
    #[must_use]
    pub fn with_accum_dtype(self, dtype: Dtype) -> Self {
        self.with_env_override(ACCUM_DTYPE.key(), accum_dtype_env_value(dtype))
    }

    /// Set `METALLIC_COMPUTE_DTYPE` using a strongly-typed dtype.
    #[must_use]
    pub fn with_compute_dtype(self, dtype: Dtype) -> Self {
        self.with_env_override(COMPUTE_DTYPE.key(), accum_dtype_env_value(dtype))
    }

    /// Return true when at least one runtime override is configured.
    #[must_use]
    pub fn has_overrides(&self) -> bool {
        !self.env_overrides.is_empty()
    }

    pub(crate) fn clone_env_overrides(&self) -> HashMap<String, String> {
        self.env_overrides.clone()
    }
}

fn accum_dtype_env_value(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::F16 => "f16",
        Dtype::BF16 => "bf16",
        Dtype::F32 => "f32",
        other => panic!(
            "Unsupported accum dtype {:?}; expected one of F16/BF16/F32 for {}",
            other,
            ACCUM_DTYPE.key()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::FoundryConfig;
    use crate::tensor::Dtype;

    #[test]
    fn with_accum_dtype_sets_env_override() {
        let cfg = FoundryConfig::default().with_accum_dtype(Dtype::F32);
        assert!(cfg.has_overrides());
    }
}
