pub mod cache;
mod interface;

use interface::{SdpaGraphInputs, SdpaGraphInterface, SdpaGraphOutput};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState};
use objc2_metal_performance_shaders::MPSCommandBuffer;

#[cfg(test)]
mod sdpa_mps_graph_test;

use cache::{CacheableMpsGraphSdpa, CacheableMpsGraphSdpaMask};

use crate::{
    CommandBuffer, Context, MetalError, Operation, Tensor, TensorElement, caching::ResourceCache, kernels::{
        DefaultKernelInvocable, GraphKernel, GraphKernelAccumulator, GraphKernelAxis, GraphKernelDtypePolicy, GraphKernelSignature, GraphKernelTensorDescriptor
    }, operation::EncoderType
};

// SDPA via MPSGraph (f16-only for now). We assume additive mask semantics: 0.0 for allowed and -inf for disallowed.
// NOTE: Mask semantics are assumed and may differ across OS versions. If causal parity issues appear in benchmarks,
// we will revisit and try alternate mask representations (e.g., boolean 0/1). Graph compilation, executable caching,
// and zero-copy bindings are handled through the resource cache so this path can expand to larger fused graphs later.

pub struct SdpaMpsGraphOp;

const SDPA_GRAPH_INPUTS: &[GraphKernelTensorDescriptor] = &[
    GraphKernelTensorDescriptor::without_notes(
        "query",
        &[GraphKernelAxis::Batch, GraphKernelAxis::SequenceQ, GraphKernelAxis::ModelDim],
    ),
    GraphKernelTensorDescriptor::without_notes(
        "key",
        &[GraphKernelAxis::Batch, GraphKernelAxis::SequenceK, GraphKernelAxis::ModelDim],
    ),
    GraphKernelTensorDescriptor::without_notes(
        "value",
        &[GraphKernelAxis::Batch, GraphKernelAxis::SequenceV, GraphKernelAxis::ModelDim],
    ),
    GraphKernelTensorDescriptor::new(
        "mask",
        &[
            GraphKernelAxis::Static(1),
            GraphKernelAxis::Static(1),
            GraphKernelAxis::SequenceQ,
            GraphKernelAxis::SequenceK,
        ],
        Some("Optional; populated when causal attention requires additive masking."),
    ),
];

const SDPA_GRAPH_OUTPUTS: &[GraphKernelTensorDescriptor] = &[GraphKernelTensorDescriptor::without_notes(
    "attention",
    &[GraphKernelAxis::Batch, GraphKernelAxis::SequenceQ, GraphKernelAxis::ModelDim],
)];

struct SdpaMpsGraphOperation<T: TensorElement> {
    // Cached MPSGraph SDPA resources
    cached_graph: CacheableMpsGraphSdpa,
    // Cached mask buffer for causal attention
    cached_mask: Option<CacheableMpsGraphSdpaMask>,
    custom_mask: Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    // Input tensors
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    // Output tensor
    out: Tensor<T>,
}

impl<T: TensorElement> Operation for SdpaMpsGraphOperation<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        // Use smart encoder management - only terminate if there's an active Metal encoder
        // This should eliminate unnecessary sync points when command buffer is already clean
        command_buffer.prepare_encoder_for_operation(EncoderType::MpsGraph)?;

        use metallic_instrumentation::GpuProfiler;
        GpuProfiler::mark_use_cpu_scope_for_cb(command_buffer.raw());

        let custom_mask_ref = self.custom_mask.as_deref();
        let graph_interface = SdpaGraphInterface::new(&self.cached_graph, self.cached_mask.as_ref(), custom_mask_ref);
        let input_bindings = graph_interface.bind_inputs(SdpaGraphInputs::from((&self.q, &self.k, &self.v)))?;
        let result_bindings = graph_interface.bind_outputs(SdpaGraphOutput::from(&self.out))?;

        // SAFETY: We re-wrap the active MTLCommandBuffer to an MPSCommandBuffer so the
        // MPSGraph executable encodes into the same timeline. The raw pointer is valid
        // for the lifetime of this encode and is owned by the caller's Context.
        let mps_command_buffer = unsafe { MPSCommandBuffer::commandBufferWithCommandBuffer(command_buffer.raw()) };

        let encode_start = std::time::Instant::now();
        let _ = unsafe {
            self.cached_graph
                .executable
                .encodeToCommandBuffer_inputsArray_resultsArray_executionDescriptor(
                    &mps_command_buffer,
                    &input_bindings,
                    Some(&*result_bindings),
                    None,
                )
        };

        let encode_elapsed = encode_start.elapsed();
        metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::InternalKernelCompleted {
            parent_op_name: "mpsgraph".to_string(),
            internal_kernel_name: "sdpa_encode".to_string(),
            duration_us: (encode_elapsed.as_secs_f64() * 1e6) as u64,
        });

        Ok(())
    }

    fn bind_kernel_args(&self, _encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        // MPSGraph operations don't bind compute encoder arguments directly - they use MPSGraph bindings
    }
}

impl DefaultKernelInvocable for SdpaMpsGraphOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a Tensor<T>, &'a Tensor<T>, bool, u32);

    fn function_id() -> Option<crate::kernels::KernelFunction> {
        None
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        // Enforce storage dtype via the GraphKernel policy.
        SdpaMpsGraphOp::validate_storage_dtype(T::DTYPE)?;

        let (q, k, v, causal, query_offset) = args;

        // Basic shape checks and preparation
        let q_dims = q.dims();
        let k_dims = k.dims();
        let v_dims = v.dims();
        if q_dims.len() != 3 || k_dims.len() != 3 || v_dims.len() != 3 {
            return Err(MetalError::InvalidShape("SDPA expects 3D tensors [B, S, D]".into()));
        }
        let (bq, sq, d_q) = (q_dims[0], q_dims[1], q_dims[2]);
        let (bk, sk, d_k) = (k_dims[0], k_dims[1], k_dims[2]);
        let (bv, sv, d_v) = (v_dims[0], v_dims[1], v_dims[2]);
        if bq != bk || bq != bv {
            return Err(MetalError::InvalidShape("SDPA batch dims must match".into()));
        }
        if d_q != d_k || d_q != d_v {
            return Err(MetalError::InvalidShape("SDPA feature dims must match (q,k,v)".into()));
        }
        if sk != sv {
            return Err(MetalError::InvalidShape("SDPA seq_k mismatch between K and V".into()));
        }

        // Trust caller - tensors are already prepared, use directly to avoid redundant preparation
        // Removed ensure_graph_ready_tensor calls and redundant ctx.prepare_tensors_for_active_cmd()
        // This eliminates ~3 micro-operations per tensor per SDPA call

        // Create output tensor (pooled)
        let out = Tensor::<T>::zeros(vec![bq, sq, d_v], ctx, false)?;

        let cache = _cache.ok_or(MetalError::ResourceCacheRequired)?;

        // Get both cached MPSGraph SDPA and mask in one call to avoid double borrowing
        let dtype_policy = SdpaMpsGraphOp::dtype_policy();
        let (cached_graph, cached_mask_entry) =
            cache.get_or_create_mpsgraph_sdpa_and_mask(bq, sq, sk, d_q, causal, T::DTYPE, dtype_policy.accumulator())?;

        let query_offset_usize = usize::try_from(query_offset)
            .map_err(|_| MetalError::InvalidShape("SDPA query offset exceeds usize on this platform".into()))?;

        let (cached_mask, custom_mask) = if !causal || query_offset_usize == 0 {
            (cached_mask_entry, None)
        } else {
            let mask_entry = cached_mask_entry
                .as_ref()
                .ok_or_else(|| MetalError::OperationFailed("Causal SDPA cache entry missing mask buffer".into()))?;

            let rows_available = mask_entry.seq_q_size;
            let stride = mask_entry.seq_k_size;
            let needed_rows = query_offset_usize
                .checked_add(sq)
                .ok_or_else(|| MetalError::InvalidShape("SDPA query offset overflow".into()))?;
            if needed_rows > rows_available {
                return Err(MetalError::OperationFailed(format!(
                    "SDPA mask cache insufficient rows for query offset {query_offset_usize} with seq_q {sq} (available {rows_available})",
                )));
            }

            let element_size = match mask_entry.data_type {
                objc2_metal_performance_shaders::MPSDataType::Float16 => core::mem::size_of::<half::f16>(),
                objc2_metal_performance_shaders::MPSDataType::Float32 => core::mem::size_of::<f32>(),
                other => {
                    return Err(MetalError::OperationFailed(format!("Unsupported SDPA mask data type: {other:?}")));
                }
            };

            let byte_offset = query_offset_usize
                .checked_mul(stride)
                .and_then(|row_elements| row_elements.checked_mul(element_size))
                .ok_or_else(|| MetalError::InvalidShape("SDPA mask offset overflow".into()))?;
            let byte_len = sq
                .checked_mul(stride)
                .and_then(|elements| elements.checked_mul(element_size))
                .ok_or_else(|| MetalError::InvalidShape("SDPA mask length overflow".into()))?;

            let alias_buffer = mask_entry.view_for(&ctx.device, byte_offset, byte_len)?;

            (None, Some(alias_buffer))
        };

        // Create the operation with all necessary data
        let operation = SdpaMpsGraphOperation {
            cached_graph,
            cached_mask,
            custom_mask,
            q: q.clone(),
            k: k.clone(),
            v: v.clone(),
            out: out.clone(),
        };

        Ok((Box::new(operation), out))
    }
}

impl GraphKernel for SdpaMpsGraphOp {
    const OP_NAME: &'static str = "sdpa_mps_graph";

    fn dtype_policy() -> GraphKernelDtypePolicy {
        GraphKernelDtypePolicy::new(
            crate::tensor::dtypes::Dtype::F16,
            GraphKernelAccumulator::Explicit(crate::tensor::dtypes::Dtype::F32),
        )
    }

    fn signature() -> GraphKernelSignature {
        GraphKernelSignature::new(SDPA_GRAPH_INPUTS, SDPA_GRAPH_OUTPUTS)
    }
}
