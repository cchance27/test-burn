use metallic_env::ACCUM_DTYPE;

use crate::{MetalError, metals::common::dtype_contract::require_uniform_dtypes, tensor::Dtype, types::TensorArg};

#[cfg(not(test))]
fn validate_flashattention_accum_env_once() -> Result<(), String> {
    use std::sync::OnceLock;

    static VALIDATION: OnceLock<Result<(), String>> = OnceLock::new();
    VALIDATION
        .get_or_init(|| match ACCUM_DTYPE.get() {
            Ok(Some(raw)) => {
                let normalized = raw.trim().to_ascii_lowercase();
                if normalized != "f32" && normalized != "float" {
                    Err(format!(
                        "requires {} to be f32|float; got '{}'. FlashAttention currently uses float softmax stats paths.",
                        ACCUM_DTYPE.key(),
                        raw
                    ))
                } else {
                    Ok(())
                }
            }
            Ok(None) => Ok(()),
            Err(err) => Err(format!("Failed reading {}: {err:#}", ACCUM_DTYPE.key())),
        })
        .clone()
}

fn require_supported_flashattention_accum_dtype(kernel_name: &str) -> Result<(), MetalError> {
    #[cfg(not(test))]
    {
        validate_flashattention_accum_env_once().map_err(|message| {
            if message.starts_with("requires ") {
                MetalError::OperationNotSupported(format!("{kernel_name} {message}"))
            } else {
                MetalError::OperationFailed(format!("{kernel_name}: {message}"))
            }
        })
    }

    #[cfg(test)]
    match ACCUM_DTYPE.get() {
        Ok(Some(raw)) => {
            let normalized = raw.trim().to_ascii_lowercase();
            if normalized != "f32" && normalized != "float" {
                return Err(MetalError::OperationNotSupported(format!(
                    "{kernel_name} requires {} to be f32|float; got '{}'. FlashAttention currently uses float softmax stats paths.",
                    ACCUM_DTYPE.key(),
                    raw
                )));
            }
            Ok(())
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
    fn flashattention_rejects_non_f32_accum_override() {
        let _guard = ACCUM_DTYPE.set_guard("bf16".to_string()).expect("set accum env");
        let err = require_supported_flashattention_accum_dtype("FlashAttentionTest").expect_err("bf16 accum should fail-fast");
        let message = format!("{err}");
        assert!(message.contains("requires METALLIC_ACCUM_DTYPE to be f32|float"));
    }
}
