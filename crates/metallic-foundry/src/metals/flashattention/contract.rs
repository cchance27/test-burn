use metallic_env::ACCUM_DTYPE;

use crate::{MetalError, metals::common::dtype_contract::require_uniform_dtypes, tensor::Dtype, types::TensorArg};

fn require_supported_flashattention_accum_dtype(kernel_name: &str) -> Result<(), MetalError> {
    match ACCUM_DTYPE.get() {
        Ok(Some(raw)) => {
            let normalized = raw.trim().to_ascii_lowercase();
            match normalized.as_str() {
                "f16" | "half" | "bf16" | "bfloat16" | "f32" | "float" => {
                    #[cfg(not(test))]
                    if normalized != "f32" && normalized != "float" {
                        use std::sync::OnceLock;
                        static WARNED_NON_F32_ACCUM: OnceLock<()> = OnceLock::new();
                        if WARNED_NON_F32_ACCUM.set(()).is_ok() {
                            tracing::warn!(
                                kernel = kernel_name,
                                accum_dtype = raw.as_str(),
                                "FlashAttention running with non-f32 accumulation; this may reduce numerical stability."
                            );
                        }
                    }
                    Ok(())
                }
                _ => Err(MetalError::OperationNotSupported(format!(
                    "{kernel_name} has invalid {}='{}'; expected one of f16|bf16|f32.",
                    ACCUM_DTYPE.key(),
                    raw
                ))),
            }
        }
        Ok(None) => Ok(()),
        Err(err) => Err(MetalError::OperationFailed(format!(
            "Failed reading {} for {kernel_name}: {err:#}",
            ACCUM_DTYPE.key()
        ))),
    }
}

pub(super) fn require_dense_tensor_contract(kernel_name: &str, tensors: &[(&str, &TensorArg)]) -> Result<(), MetalError> {
    require_supported_flashattention_accum_dtype(kernel_name)?;

    let mut first: Option<(&str, Dtype)> = None;
    for (name, tensor) in tensors {
        let dtype = tensor.dtype;
        if !matches!(dtype, Dtype::F16 | Dtype::F32) {
            return Err(MetalError::OperationNotSupported(format!(
                "{kernel_name} supports only dense F16/F32 {name} tensor dtype, got {:?}.",
                dtype
            )));
        }
        if let Some((first_name, first_dtype)) = first {
            if dtype != first_dtype {
                return require_uniform_dtypes(kernel_name, &[(first_name, first_dtype), (*name, dtype)]).map(|_| ());
            }
        } else {
            first = Some((*name, dtype));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use metallic_env::ACCUM_DTYPE;

    use super::require_supported_flashattention_accum_dtype;

    #[test]
    #[serial_test::serial]
    fn flashattention_accepts_f32_accum_override() {
        let _guard = ACCUM_DTYPE.set_guard("f32".to_string()).expect("set accum env");
        require_supported_flashattention_accum_dtype("FlashAttentionTest").expect("f32 accum should be accepted");
    }

    #[test]
    #[serial_test::serial]
    fn flashattention_accepts_non_f32_accum_override() {
        let _guard = ACCUM_DTYPE.set_guard("bf16".to_string()).expect("set accum env");
        require_supported_flashattention_accum_dtype("FlashAttentionTest").expect("bf16 accum should be accepted");
    }

    #[test]
    #[serial_test::serial]
    fn flashattention_rejects_invalid_accum_override() {
        let _guard = ACCUM_DTYPE.set_guard("invalid".to_string()).expect("set accum env");
        let err = require_supported_flashattention_accum_dtype("FlashAttentionTest").expect_err("invalid accum should fail-fast");
        let message = format!("{err}");
        assert!(message.contains("expected one of f16|bf16|f32"), "unexpected error: {message}");
    }
}
