use super::{Context, MetalError, Operation, Tensor, resource_cache::ResourceCache};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder as _, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice as _, MTLLibrary as _, MTLSize,
};

use crate::metallic::encoder::{
    dispatch_threadgroups, set_buffer, set_bytes, set_compute_pipeline_state,
};

/// Ensure the RoPE compute pipeline is compiled and cached on the Context.
/// RoPE (rotary positional embeddings) expects:
///  - buffer(0): device float* input
///  - buffer(1): device float* output
///  - buffer(2): device float* cos  (shape: [seq_len, dim/2])
///  - buffer(3): device float* sin  (shape: [seq_len, dim/2])
///  - buffer(4): constant uint& dim
///  - buffer(5): constant uint& seq_len
///  - buffer(6): constant uint& total_elements
///
/// The kernel treats the last dimension `dim` as interleaved pairs:
/// for pair p (0..dim/2):
///   i = 2*p, j = 2*p+1
///   x_i' = x_i * cos[pos, p] - x_j * sin[pos, p]
///   x_j' = x_j * cos[pos, p] + x_i * sin[pos, p]
pub fn ensure_rope_pipeline(ctx: &mut Context) -> Result<(), MetalError> {
    if ctx.rope_pipeline.is_some() {
        return Ok(());
    }

    let source = r#"
    using namespace metal;

    kernel void rope_kernel(device float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            device float* cos_buf [[buffer(2)]],
                            device float* sin_buf [[buffer(3)]],
                            constant uint& dim [[buffer(4)]],
                            constant uint& seq_len [[buffer(5)]],
                            constant uint& total_elements [[buffer(6)]],
                            uint gid [[thread_position_in_grid]]) {
        if (gid >= total_elements) return;

        // Determine row (sequence position) and feature index within row
        uint feature_idx = gid % dim;
        uint row_idx = gid / dim;

        // Position in sequence (assume rows are arranged so that seq dimension varies fastest across rows)
        uint pos = row_idx % seq_len;

        // Half-split RoPE: pair indices across the two halves of the last dimension
        uint half_dim = dim / 2u;
        uint pair = (feature_idx < half_dim) ? feature_idx : (feature_idx - half_dim);
        float cosv = cos_buf[pos * half_dim + pair];
        float sinv = sin_buf[pos * half_dim + pair];

        if (feature_idx < half_dim) {
            // first half element x_i pairs with x_j at index +half_dim
            float x_i = input[gid];
            float x_j = input[row_idx * dim + feature_idx + half_dim];
            float out_i = x_i * cosv - x_j * sinv;
            output[gid] = out_i;
        } else {
            // second half element x_j pairs with x_i at index -half_dim
            float x_j = input[gid];
            float x_i = input[row_idx * dim + (feature_idx - half_dim)];
            float out_j = x_j * cosv + x_i * sinv;
            output[gid] = out_j;
        }
    }
    "#;

    let source_ns = NSString::from_str(source);
    let library = ctx
        .device
        .newLibraryWithSource_options_error(&source_ns, None)
        .map_err(|err| MetalError::LibraryCompilationFailed(err.to_string()))?;

    let fn_name = NSString::from_str("rope_kernel");
    let function = library
        .newFunctionWithName(&fn_name)
        .ok_or_else(|| MetalError::FunctionCreationFailed(fn_name.to_string()))?;
    let pipeline = ctx
        .device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|_err| MetalError::PipelineCreationFailed)?;

    ctx.rope_pipeline = Some(pipeline);
    Ok(())
}

/// Operation to apply RoPE to an input tensor.
pub struct RoPE {
    pub input: Tensor,
    pub output: Tensor,
    pub cos: Tensor,
    pub sin: Tensor,
    pub dim: u32,
    pub seq_len: u32,
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl RoPE {
    pub fn new(
        input: Tensor,
        output: Tensor,
        cos: Tensor,
        sin: Tensor,
        dim: u32,
        seq_len: u32,
        pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    ) -> Result<Self, MetalError> {
        // Basic validation
        if dim == 0 || !dim.is_multiple_of(2) {
            return Err(MetalError::InvalidShape(format!(
                "dim must be positive and even for RoPE, got {}",
                dim
            )));
        }
        if input.dims() != output.dims() {
            return Err(MetalError::InvalidShape(format!(
                "input and output shapes must match: input={:?} output={:?}",
                input.dims(),
                output.dims()
            )));
        }
        // cos/sin should be [seq_len, dim/2]
        if cos.dims() != [seq_len as usize, (dim as usize) / 2] {
            return Err(MetalError::InvalidShape(format!(
                "cos shape {:?} does not match [seq_len, dim/2] = [{}, {}]",
                cos.dims(),
                seq_len,
                dim / 2
            )));
        }
        if sin.dims() != [seq_len as usize, (dim as usize) / 2] {
            return Err(MetalError::InvalidShape(format!(
                "sin shape {:?} does not match [seq_len, dim/2] = [{}, {}]",
                sin.dims(),
                seq_len,
                dim / 2
            )));
        }

        Ok(Self {
            input,
            output,
            cos,
            sin,
            dim,
            seq_len,
            pipeline,
        })
    }
}

impl Operation for RoPE {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        let total_elements = self.input.len() as u32;
        let threads_per_tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let groups = MTLSize {
            width: total_elements.div_ceil(256) as usize,
            height: 1,
            depth: 1,
        };

        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.input.buf, self.input.offset);
        set_buffer(&encoder, 1, &self.output.buf, self.output.offset);
        set_buffer(&encoder, 2, &self.cos.buf, self.cos.offset);
        set_buffer(&encoder, 3, &self.sin.buf, self.sin.offset);
        set_bytes(&encoder, 4, &self.dim);
        set_bytes(&encoder, 5, &self.seq_len);
        set_bytes(&encoder, 6, &total_elements);

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}
