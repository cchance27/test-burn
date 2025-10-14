use crate::{
    cache_keys::MpsMatrixDescriptorKey,
    context::Context,
    error::MetalError,
    kernels::softmax_kernel::SoftmaxKernelOp,
    resource_cache::ResourceCache,
    tensor::{Dtype, MpsMatrixBatchView, Tensor, TensorElement},
};
use metallic_env::SOFTMAX_BACKEND_VAR;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SoftmaxBackendPreference {
    Auto,
    KernelOnly,
    MpsOnly,
}

impl SoftmaxBackendPreference {
    fn forces_kernel(self) -> bool {
        matches!(self, Self::KernelOnly)
    }
}

static BACKEND_PREFERENCE: OnceLock<SoftmaxBackendPreference> = OnceLock::new();

pub fn softmax_backend_preference() -> SoftmaxBackendPreference {
    *BACKEND_PREFERENCE.get_or_init(|| {
        let raw = SOFTMAX_BACKEND_VAR
            .get()
            .unwrap_or(None)
            .unwrap_or_else(|| "auto".to_string())
            .to_lowercase();
        match raw.trim() {
            "kernel" | "compute" | "pipeline" => SoftmaxBackendPreference::KernelOnly,
            "mps" | "metal" => SoftmaxBackendPreference::MpsOnly,
            "auto" | "" | "default" => SoftmaxBackendPreference::Auto,
            other => {
                eprintln!("Unknown {} value '{other}', falling back to auto", SOFTMAX_BACKEND_VAR.key());
                SoftmaxBackendPreference::Auto
            }
        }
    })
}

#[allow(clippy::too_many_arguments)]
pub fn apply_softmax<T: TensorElement>(
    ctx: &mut Context<T>,
    mut cache: Option<&mut ResourceCache>,
    attn: &Tensor<T>,
    batch: usize,
    rows: usize,
    columns: usize,
    causal: bool,
    query_offset: u32,
    allow_mps: bool,
) -> Result<Tensor<T>, MetalError> {
    let view = attn.as_mps_matrix_batch_view()?;

    if view.rows != rows || view.columns != columns {
        return Err(MetalError::InvalidShape(format!(
            "Attention matrix dimensions {:?} don't match expected {} x {}",
            attn.dims(),
            rows,
            columns
        )));
    }

    if view.batch != batch {
        return Err(MetalError::InvalidShape(format!(
            "Attention batch dimension {} does not match expected {}",
            view.batch, batch
        )));
    }

    let rows_total = batch * rows;
    let dtype = attn.dtype;

    let preference = softmax_backend_preference();
    let supports_mps_dtype = matches!(dtype, Dtype::F32 | Dtype::F16);
    let can_use_mps = allow_mps && supports_mps_dtype && !causal && query_offset == 0 && !preference.forces_kernel();
    if can_use_mps && let Some(cache_slot) = cache.as_mut() {
        let cache_ref: &mut ResourceCache = cache_slot;
        try_apply_mps_softmax(ctx, cache_ref, attn, &view, batch, rows, columns, dtype, causal)?;
        return Ok(attn.clone());
    }

    let result = match cache {
        Some(cache_ref) => ctx.call_with_cache::<SoftmaxKernelOp>(
            (attn, rows_total as u32, rows as u32, columns as u32, causal as u32, query_offset),
            cache_ref,
        )?,
        None => ctx.call::<SoftmaxKernelOp>((attn, rows_total as u32, rows as u32, columns as u32, causal as u32, query_offset))?,
    };
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn try_apply_mps_softmax<T: TensorElement>(
    ctx: &mut Context<T>,
    cache: &mut ResourceCache,
    attn: &Tensor<T>,
    view: &MpsMatrixBatchView,
    batch: usize,
    rows: usize,
    columns: usize,
    dtype: Dtype,
    causal: bool,
) -> Result<(), MetalError> {
    ctx.prepare_tensors_for_active_cmd(&[attn])?;

    let descriptor_key = MpsMatrixDescriptorKey {
        rows,
        columns,
        row_bytes: view.row_bytes,
        matrices: view.batch,
        matrix_bytes: view.matrix_bytes,
        dtype,
    };
    let descriptor = cache.get_or_create_descriptor(descriptor_key, &ctx.device)?;
    let softmax = cache.get_or_create_softmax_full(rows, columns, dtype, causal, &ctx.device)?;

    let command_buffer = ctx.active_command_buffer_mut_without_cache()?;
    let op = crate::kernels::softmax_mps::create_softmax_mps_operation_from_context(attn.clone(), descriptor, softmax, batch);
    command_buffer.record(&op, cache)?;
    ctx.mark_tensor_pending(attn);
    ctx.finalize_active_command_buffer_if_latency();
    Ok(())
}
