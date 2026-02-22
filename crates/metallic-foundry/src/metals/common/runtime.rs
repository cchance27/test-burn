use crate::{MetalError, spec::TensorBindings, types::TensorArg};

#[inline]
pub fn resolve_batch(bindings: &TensorBindings) -> u32 {
    bindings.get_int_global("m").unwrap_or(1).max(1) as u32
}

#[inline]
pub fn require_non_empty_io(op: &'static str, input: &TensorArg, output: &TensorArg) -> Result<(), MetalError> {
    if input.dims.is_empty() || output.dims.is_empty() {
        return Err(MetalError::InvalidShape(format!("{op} expects non-empty input/output shapes")));
    }
    Ok(())
}

#[inline]
pub fn tail_dim(tensor: &TensorArg) -> Result<u32, MetalError> {
    tensor
        .dims
        .last()
        .copied()
        .map(|v| v as u32)
        .ok_or_else(|| MetalError::InvalidShape("expected non-empty tensor shape".into()))
}

#[inline]
pub fn require_vector_len(name: &'static str, tensor: &TensorArg, expected: u32) -> Result<(), MetalError> {
    if tensor.dims.len() != 1 || tensor.dims[0] as u32 != expected {
        return Err(MetalError::DimensionMismatch {
            expected: expected as usize,
            actual: tensor.dims.iter().product(),
        });
    }
    let _ = name;
    Ok(())
}

#[inline]
pub fn require_contiguous_last_dim(name: &'static str, strides: &[usize]) -> Result<(), MetalError> {
    if strides.last().copied() != Some(1) {
        return Err(MetalError::OperationNotSupported(format!(
            "Prefill requires contiguous last dim for {name}, got strides={strides:?}"
        )));
    }
    Ok(())
}
