use serde::{Deserialize, Serialize};

use crate::{
    MetalError, foundry::{
        Foundry, spec::{DynamicValue, Ref, Step, TensorBindings}
    }, metals::sdpa::scaled_dot_product_attention, types::{KernelArg, TensorArg}
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

/// Shared SDPA execution logic used by both Step::execute and CompiledStep::execute.
fn execute_sdpa(
    foundry: &mut Foundry,
    q_arg: &TensorArg,
    k_arg: &TensorArg,
    v_arg: &TensorArg,
    output_arg: &TensorArg,
    causal: bool,
    kv_head_major: bool,
    n_heads: Option<u32>,
    head_dim: Option<u32>,
    query_offset: u32,
    kv_seq: Option<u32>,
    max_seq: Option<u32>,
) -> Result<(), MetalError> {
    // Helper to construct Tensor view, optionally reshaping.
    let make_view = |arg: &dyn KernelArg, seq_len: u32, head_major: bool| {
        let mut dims = arg.dims().to_vec();
        let mut strides = arg.strides().to_vec();

        // If reshaping requested (MHA mode)
        if let (Some(heads), Some(hdim)) = (n_heads, head_dim) {
            dims = vec![heads as usize, seq_len as usize, hdim as usize];
            if head_major {
                // Head-major layout: [heads, seq, hdim]
                let capacity_seq = max_seq.unwrap_or(seq_len);
                let head_stride = (capacity_seq * hdim) as usize;
                strides = vec![head_stride, hdim as usize, 1];
            } else {
                // Seq-major layout: [seq, heads, hdim]
                strides = vec![hdim as usize, heads as usize * hdim as usize, 1];
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
    let q = make_view(q_arg, q_seq, true);

    // K/V are interpreted as [heads, seq, dim]
    let kv_seq_val = kv_seq.unwrap_or(1);
    let k = make_view(k_arg, kv_seq_val, kv_head_major);
    let v = make_view(v_arg, kv_seq_val, kv_head_major);

    // Execute SDPA
    let result = scaled_dot_product_attention(foundry, &q, &k, &v, causal, query_offset)?;

    // Copy result to output buffer
    let size_bytes = result.dims().iter().product::<usize>() * result.dtype().size_bytes();

    foundry.blit_copy(
        result.buffer(),
        result.offset(),
        output_arg.buffer(),
        output_arg.offset(),
        size_bytes,
    )?;

    Ok(())
}

#[typetag::serde(name = "Sdpa")]
impl Step for SdpaStep {
    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        let q_arg = bindings.resolve(&self.q)?;
        let k_arg = bindings.resolve(&self.k)?;
        let v_arg = bindings.resolve(&self.v)?;
        let output_arg = bindings.resolve(&self.output)?;

        execute_sdpa(
            foundry,
            &q_arg,
            &k_arg,
            &v_arg,
            &output_arg,
            self.causal,
            self.kv_head_major,
            self.n_heads,
            self.head_dim,
            self.query_offset.resolve(bindings),
            self.kv_seq_len.as_ref().map(|v| v.resolve(bindings)),
            self.max_seq_len.as_ref().map(|v| v.resolve(bindings)),
        )
    }

    fn name(&self) -> &'static str {
        "Sdpa"
    }

    fn compile(
        &self,
        resolver: &mut TensorBindings,
        symbols: &mut crate::foundry::spec::SymbolTable,
    ) -> Vec<Box<dyn crate::foundry::spec::CompiledStep>> {
        let q_idx = symbols.get_or_create(resolver.interpolate(self.q.0.clone()));
        let k_idx = symbols.get_or_create(resolver.interpolate(self.k.0.clone()));
        let v_idx = symbols.get_or_create(resolver.interpolate(self.v.0.clone()));
        let output_idx = symbols.get_or_create(resolver.interpolate(self.output.0.clone()));

        vec![Box::new(CompiledSdpaStep {
            q_idx,
            k_idx,
            v_idx,
            output_idx,
            causal: self.causal,
            kv_head_major: self.kv_head_major,
            query_offset: self.query_offset.clone(),
            kv_seq_len: self.kv_seq_len.clone(),
            max_seq_len: self.max_seq_len.clone(),
            n_heads: self.n_heads,
            head_dim: self.head_dim,
        })]
    }
}

#[derive(Debug)]
pub struct CompiledSdpaStep {
    pub q_idx: usize,
    pub k_idx: usize,
    pub v_idx: usize,
    pub output_idx: usize,
    pub causal: bool,
    pub kv_head_major: bool,
    pub query_offset: DynamicValue<u32>,
    pub kv_seq_len: Option<DynamicValue<u32>>,
    pub max_seq_len: Option<DynamicValue<u32>>,
    pub n_heads: Option<u32>,
    pub head_dim: Option<u32>,
}

impl crate::foundry::spec::CompiledStep for CompiledSdpaStep {
    fn execute(
        &self,
        foundry: &mut Foundry,
        fast_bindings: &crate::foundry::spec::FastBindings,
        bindings: &TensorBindings,
    ) -> Result<(), MetalError> {
        let q_arg = fast_bindings
            .get(self.q_idx)
            .ok_or_else(|| MetalError::InvalidShape(format!("Q tensor not found at idx {}", self.q_idx)))?;
        let k_arg = fast_bindings
            .get(self.k_idx)
            .ok_or_else(|| MetalError::InvalidShape(format!("K tensor not found at idx {}", self.k_idx)))?;
        let v_arg = fast_bindings
            .get(self.v_idx)
            .ok_or_else(|| MetalError::InvalidShape(format!("V tensor not found at idx {}", self.v_idx)))?;
        let output_arg = fast_bindings
            .get(self.output_idx)
            .ok_or_else(|| MetalError::InvalidShape(format!("Output tensor not found at idx {}", self.output_idx)))?;

        execute_sdpa(
            foundry,
            q_arg,
            k_arg,
            v_arg,
            output_arg,
            self.causal,
            self.kv_head_major,
            self.n_heads,
            self.head_dim,
            self.query_offset.resolve(bindings),
            self.kv_seq_len.as_ref().map(|v| v.resolve(bindings)),
            self.max_seq_len.as_ref().map(|v| v.resolve(bindings)),
        )
    }
}
