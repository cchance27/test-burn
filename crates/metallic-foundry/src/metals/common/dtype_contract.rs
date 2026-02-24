use crate::{
    MetalError, policy::{active_policy_variant, resolve_policy_detailed}, tensor::Dtype
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelDtypeDescriptor {
    pub source: Dtype,
    pub storage: Dtype,
    pub compute: Dtype,
    pub accum: Dtype,
    pub source_size_bytes: usize,
    pub storage_size_bytes: usize,
    pub compute_size_bytes: usize,
    pub accum_size_bytes: usize,
    pub lossy_cast: bool,
}

impl KernelDtypeDescriptor {
    pub fn from_source_dtype(dtype: Dtype) -> Result<Self, MetalError> {
        let resolution = resolve_policy_detailed(dtype, active_policy_variant())
            .map_err(|err| MetalError::OperationFailed(format!("policy resolution failed for {dtype:?}: {err:#}")))?
            .resolution;
        let accum = Dtype::F32;
        Ok(Self {
            source: resolution.source_dtype,
            storage: resolution.storage_dtype,
            compute: resolution.compute_dtype,
            accum,
            source_size_bytes: resolution.source_dtype.size_bytes(),
            storage_size_bytes: resolution.storage_dtype.size_bytes(),
            compute_size_bytes: resolution.compute_dtype.size_bytes(),
            accum_size_bytes: accum.size_bytes(),
            lossy_cast: resolution.lossy_cast,
        })
    }

    #[must_use]
    pub fn simd_lanes_for_bytes(self, vector_bytes: usize) -> usize {
        let elem = self.storage_size_bytes.max(1);
        (vector_bytes / elem).max(1)
    }
}

/// Enforce mixed-policy fail-fast across multi-input kernels.
pub fn require_uniform_dtypes(kernel_name: &str, tensors: &[(&str, Dtype)]) -> Result<Dtype, MetalError> {
    let first = tensors
        .first()
        .ok_or_else(|| MetalError::OperationFailed(format!("{kernel_name} requires at least one tensor dtype")))?;
    let expected = first.1;
    if tensors.iter().all(|(_, dtype)| *dtype == expected) {
        return Ok(expected);
    }

    let details = tensors
        .iter()
        .map(|(name, dtype)| format!("{name}={dtype:?}/{}B", dtype.size_bytes()))
        .collect::<Vec<_>>()
        .join(", ");
    Err(MetalError::OperationFailed(format!(
        "{kernel_name} mixed-policy is unsupported ({details})."
    )))
}

#[cfg(test)]
mod tests {
    use metallic_env::POLICY_VARIANT;

    use super::{KernelDtypeDescriptor, require_uniform_dtypes};
    use crate::tensor::Dtype;

    #[test]
    fn uniform_dtype_contract_rejects_mixed() {
        let err = require_uniform_dtypes("UnitKernel", &[("a", Dtype::F16), ("b", Dtype::F32)]).expect_err("must fail");
        let msg = format!("{err}");
        assert!(msg.contains("mixed-policy is unsupported"), "unexpected message: {msg}");
        assert!(msg.contains("a=F16/2B"), "expected size detail: {msg}");
        assert!(msg.contains("b=F32/4B"), "expected size detail: {msg}");
    }

    #[test]
    fn descriptor_resolves_f32_preserve() {
        let _guard = POLICY_VARIANT
            .set_guard("preserve:f16".to_string())
            .expect("set METALLIC_POLICY_VARIANT");
        let desc = KernelDtypeDescriptor::from_source_dtype(Dtype::F32).expect("resolve descriptor");
        assert_eq!(desc.source, Dtype::F32);
        assert_eq!(desc.storage, Dtype::F32);
        assert_eq!(desc.compute, Dtype::F32);
        assert_eq!(desc.accum, Dtype::F32);
        assert_eq!(desc.source_size_bytes, 4);
        assert_eq!(desc.storage_size_bytes, 4);
        assert_eq!(desc.simd_lanes_for_bytes(16), 4);
    }
}
