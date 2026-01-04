use objc2_metal::{MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
use serde::{Deserialize, Serialize};

use crate::{
    MetalError, foundry::{
        Foundry, spec::{DynamicValue, Ref, Step, TensorBindings}
    }, metals::sdpa::scaled_dot_product_attention, types::KernelArg
};

/// DSL Step for Scaled Dot Product Attention.
#[derive(Debug, Serialize, Deserialize)]
pub struct SdpaStep {
    pub q: Ref,
    pub k: Ref,
    pub v: Ref,
    pub output: Ref,
    pub causal: bool,
    #[serde(default)]
    pub query_offset: DynamicValue<u32>,
    // Optional reshaping params for K/V (needed if they are flat buffers)
    #[serde(default)]
    pub kv_seq_len: Option<DynamicValue<u32>>,
    #[serde(default)]
    pub n_heads: Option<u32>,
    #[serde(default)]
    pub head_dim: Option<u32>,
    /// When true, interpret K/V as head-major layout: [heads, seq, hdim].
    /// Default is false (sequence-major view) to preserve existing behavior.
    #[serde(default)]
    pub kv_head_major: bool,
    #[serde(default)]
    pub max_seq_len: Option<DynamicValue<u32>>,
}

#[typetag::serde(name = "Sdpa")]
impl Step for SdpaStep {
    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        let q_arg = bindings.resolve(&self.q)?;
        let k_arg = bindings.resolve(&self.k)?;
        let v_arg = bindings.resolve(&self.v)?;

        // Resolve dynamic params
        let query_offset = self.query_offset.resolve(bindings);
        let kv_seq = self.kv_seq_len.as_ref().map(|v| v.resolve(bindings));
        let max_seq = self.max_seq_len.as_ref().map(|v| v.resolve(bindings));

        // Helper to construct Tensor view, optionally reshaping.
        // Note: Q from KvRearrange is head-major: [heads, seq, hdim] (with batch folded into heads).
        // K/V from RepeatKvHeads are also head-major: [heads, seq, hdim].
        let make_view = |arg: &dyn KernelArg, seq_len: u32, head_major: bool| {
            let mut dims = arg.dims().to_vec();
            let mut strides = arg.strides().to_vec();

            // If reshaping requested (MHA mode)
            if let (Some(heads), Some(hdim)) = (self.n_heads, self.head_dim) {
                dims = vec![heads as usize, seq_len as usize, hdim as usize];
                if head_major {
                    // Head-major layout: [heads, seq, hdim]
                    // Use max_seq_len for stride if provided (for pre-allocated buffers), else packed
                    // Stride is number of elements to skip to get to Next Head
                    // If buffer is [Head0...MaxSeq...Head1...], stride = MaxSeq * HeadDim
                    let capacity_seq = max_seq.unwrap_or(seq_len);
                    let head_stride = (capacity_seq * hdim) as usize;
                    strides = vec![
                        head_stride,   // head stride
                        hdim as usize, // seq stride
                        1,             // hdim stride
                    ];
                } else {
                    // Seq-major layout: [seq, heads, hdim]
                    strides = vec![
                        hdim as usize,                  // head stride
                        heads as usize * hdim as usize, // seq stride
                        1,                              // hdim stride
                    ];
                }
            }

            crate::foundry::tensor::Tensor::<crate::tensor::dtypes::F16, crate::foundry::storage::View>::from_raw_parts(
                arg.buffer().clone(),
                dims,
                strides,
                arg.offset(),
            )
        };

        // Q sequence length is determined by the input tensor
        let q_dims = q_arg.dims();
        let q_seq = if q_dims.len() >= 2 { q_dims[1] as u32 } else { 1 };
        let q = make_view(&q_arg, q_seq, true);

        // K/V are interpreted as [heads, seq, dim]
        let kv_seq_val = kv_seq.unwrap_or(1); // Default to 1 if not dynamic
        let k = make_view(&k_arg, kv_seq_val, self.kv_head_major);
        let v = make_view(&v_arg, kv_seq_val, self.kv_head_major);

        // Execute SDPA
        let result = scaled_dot_product_attention(foundry, &q, &k, &v, self.causal, query_offset)?;

        // Copy result to output buffer
        let output_arg = bindings.resolve(&self.output)?;

        let cmd = foundry
            .queue
            .commandBuffer()
            .ok_or(MetalError::OperationFailed("No command buffer".into()))?;
        let blit = cmd
            .blitCommandEncoder()
            .ok_or(MetalError::OperationFailed("No blit encoder".into()))?;

        unsafe {
            // Calculate size based on result dimensions
            let size_bytes = result.dims().iter().product::<usize>() * result.dtype().size_bytes();

            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                result.buffer(),
                result.offset(),
                output_arg.buffer(),
                output_arg.offset(),
                size_bytes,
            );
        }
        blit.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        Ok(())
    }

    fn name(&self) -> &'static str {
        "Sdpa"
    }
}
