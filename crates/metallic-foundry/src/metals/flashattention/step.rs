use serde::{Deserialize, Serialize};

use super::{
    kernels::{RopeFlashDecodeArgs, get_rope_flash_decode_kernel}, variants::{FlashDecodeVariant, flash_decode_variant_from_env, select_flash_decode_variant_m2m3}
};
use crate::{
    Foundry, MetalError, metals::{flashattention::stages::SdpaParams, rope::RopeParamsResolved}, spec::{CompiledStep, DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}, types::{
        TensorArg, dispatch::{DispatchConfig, GridSize, ThreadgroupSize}
    }
};

/// Utility wrapper for emitting/dispatching the fused RoPE→SDPA *decode* kernel.
///
/// This is primarily used for diagnostics (e.g. dumping the generated Metal source). It is not the
/// main SDPA dispatch path used by the model DSL (`metals::sdpa::step`).
pub struct RopeFlashDecodeStep;

impl RopeFlashDecodeStep {
    /// Get the generated Metal source code for this kernel.
    pub fn source() -> String {
        // Default to the primary Qwen2.5-ish decode shape.
        let head_dim = 64u32;
        let kv_len = 1024u32;
        let variant = flash_decode_variant_from_env(head_dim)
            .ok()
            .flatten()
            .unwrap_or_else(|| select_flash_decode_variant_m2m3(head_dim, kv_len));
        let kernel = get_rope_flash_decode_kernel(head_dim, variant);
        match crate::Kernel::source(&*kernel) {
            crate::KernelSource::String(s) => s,
            _ => "Source not available (Binary/Other)".to_string(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compile(
        foundry: &mut Foundry,
        q: &TensorArg,
        k: &TensorArg,
        v: &TensorArg,
        cos: &TensorArg,
        sin: &TensorArg,
        output: &TensorArg,
        rope_params: RopeParamsResolved,
        sdpa_params: SdpaParams,
        batch: u32,
        heads: u32,
        head_dim: u32,
        q_strides: (u32, u32),
        k_strides: (u32, u32),
        v_strides: (u32, u32),
        out_strides: (u32, u32),
    ) -> Result<CompiledRopeFlashDecodeStep, MetalError> {
        let variant =
            flash_decode_variant_from_env(head_dim)?.unwrap_or_else(|| select_flash_decode_variant_m2m3(head_dim, sdpa_params.kv_len));
        Self::compile_with_variant(
            foundry,
            q,
            k,
            v,
            cos,
            sin,
            output,
            rope_params,
            sdpa_params,
            batch,
            heads,
            head_dim,
            variant,
            q_strides,
            k_strides,
            v_strides,
            out_strides,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compile_with_variant(
        _foundry: &mut Foundry,
        q: &TensorArg,
        k: &TensorArg,
        v: &TensorArg,
        cos: &TensorArg,
        sin: &TensorArg,
        output: &TensorArg,
        rope_params: RopeParamsResolved,
        sdpa_params: SdpaParams,
        batch: u32,
        heads: u32,
        head_dim: u32,
        variant: FlashDecodeVariant,
        q_strides: (u32, u32),
        k_strides: (u32, u32),
        v_strides: (u32, u32),
        out_strides: (u32, u32),
    ) -> Result<CompiledRopeFlashDecodeStep, MetalError> {
        variant.validate_for_head_dim(head_dim)?;

        // Ensure kernel is initialized
        let kernel = get_rope_flash_decode_kernel(head_dim, variant);
        let source = match crate::Kernel::source(&*kernel) {
            crate::KernelSource::String(s) => s,
            _ => "N/A".to_string(),
        };

        Ok(CompiledRopeFlashDecodeStep {
            source,
            q: q.clone(),
            k: k.clone(),
            v: v.clone(),
            cos: cos.clone(),
            sin: sin.clone(),
            output: output.clone(),
            params_rope: rope_params,
            sdpa_params,
            batch,
            heads,
            head_dim,
            variant,
            q_strides,
            k_strides,
            v_strides,
            out_strides,
        })
    }
}

#[derive(Debug)]
pub struct CompiledRopeFlashDecodeStep {
    pub source: String,
    q: TensorArg,
    k: TensorArg,
    v: TensorArg,
    cos: TensorArg,
    sin: TensorArg,
    output: TensorArg,

    params_rope: RopeParamsResolved,
    sdpa_params: SdpaParams,

    variant: FlashDecodeVariant,

    q_strides: (u32, u32),
    k_strides: (u32, u32),
    v_strides: (u32, u32),
    out_strides: (u32, u32),

    batch: u32,
    heads: u32,
    head_dim: u32,
}

impl CompiledRopeFlashDecodeStep {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q: TensorArg,
        k: TensorArg,
        v: TensorArg,
        cos: TensorArg,
        sin: TensorArg,
        output: TensorArg,
        params_rope: RopeParamsResolved,
        sdpa_params: SdpaParams,
        variant: FlashDecodeVariant,
        batch: u32,
        heads: u32,
        head_dim: u32,
        q_strides: (u32, u32),
        k_strides: (u32, u32),
        v_strides: (u32, u32),
        out_strides: (u32, u32),
        source: String,
    ) -> Self {
        Self {
            source,
            q,
            k,
            v,
            cos,
            sin,
            output,
            params_rope,
            sdpa_params,
            variant,
            batch,
            heads,
            head_dim,
            q_strides,
            k_strides,
            v_strides,
            out_strides,
        }
    }
}

impl CompiledStep for CompiledRopeFlashDecodeStep {
    fn execute(
        &self,
        foundry: &mut Foundry,
        _fast_bindings: &FastBindings,
        _globals: &TensorBindings,
        _symbols: &SymbolTable,
    ) -> Result<(), MetalError> {
        // Construct Args
        let args = RopeFlashDecodeArgs {
            q: self.q.clone(),
            k: self.k.clone(),
            v: self.v.clone(),
            output: self.output.clone(),
            q_stride_b: self.q_strides.0,
            q_stride_h: self.q_strides.1,
            k_stride_b: self.k_strides.0,
            k_stride_h: self.k_strides.1,
            v_stride_b: self.v_strides.0,
            v_stride_h: self.v_strides.1,
            out_stride_b: self.out_strides.0,
            out_stride_h: self.out_strides.1,
            cos: self.cos.clone(),
            sin: self.sin.clone(),
            params_rope: self.params_rope,
            sdpa_params: self.sdpa_params,
        };
        // Dispatch:
        // - Grid: (1, Heads, Batch) ⇒ one threadgroup per (head, batch).
        // - For the FlashAttention-style decode kernels we use multiple simdgroups per threadgroup
        //   to cover a large KV block per iteration and reduce partials efficiently.
        let grid = GridSize::new(1, self.heads as usize, self.batch as usize);
        let group = ThreadgroupSize::d1(self.variant.threads_per_tg() as usize);
        let config = DispatchConfig::new(grid, group);

        let kernel = get_rope_flash_decode_kernel(self.head_dim, self.variant);
        let bound = kernel.clone().bind_arc(args, config);
        foundry.run(&bound)
    }

    fn name(&self) -> &'static str {
        "RopeFlashDecode"
    }
}

// =============================================================================
// DSL-Compatible FlashDecodeKernel (standalone online SDPA; decode-like)
// =============================================================================

/// Flash Decode Step for DSL compatibility.
/// Standalone online SDPA (streaming/online-softmax). Decode-oriented (M=1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlashDecodeKernel {
    pub q: Ref,
    pub k: Ref,
    pub v: Ref,
    pub output: Ref,
    #[serde(default)]
    pub causal: bool,
    pub n_heads: DynamicValue<u32>,
    pub head_dim: DynamicValue<u32>,
    pub kv_seq_len: DynamicValue<u32>,
    #[serde(default)]
    pub query_offset: DynamicValue<u32>,
    #[serde(default)]
    pub kv_head_major: bool,
}

#[derive(Debug, Clone)]
pub struct CompiledFlashDecodeKernel {
    pub step: FlashDecodeKernel,
    pub q_idx: usize,
    pub k_idx: usize,
    pub v_idx: usize,
    pub output_idx: usize,
}

pub use super::runtime::{run_flash_decode, run_flash_decode_with_variant};

#[typetag::serde(name = "FlashDecode")]
impl Step for FlashDecodeKernel {
    fn name(&self) -> &'static str {
        "FlashDecode"
    }

    fn execute(&self, _foundry: &mut Foundry, _bindings: &mut TensorBindings) -> Result<(), MetalError> {
        Err(MetalError::OperationNotSupported("FlashDecode only supports compile()".into()))
    }

    fn compile(&self, bindings: &mut TensorBindings, symbols: &mut SymbolTable) -> Vec<Box<dyn CompiledStep>> {
        let q_idx = symbols.get_or_create(bindings.interpolate(self.q.0.clone()));
        let k_idx = symbols.get_or_create(bindings.interpolate(self.k.0.clone()));
        let v_idx = symbols.get_or_create(bindings.interpolate(self.v.0.clone()));
        let output_idx = symbols.get_or_create(bindings.interpolate(self.output.0.clone()));

        vec![Box::new(CompiledFlashDecodeKernel {
            step: self.clone(),
            q_idx,
            k_idx,
            v_idx,
            output_idx,
        })]
    }
}

impl CompiledStep for CompiledFlashDecodeKernel {
    fn execute(
        &self,
        foundry: &mut Foundry,
        fast_bindings: &FastBindings,
        bindings: &TensorBindings,
        _symbols: &SymbolTable,
    ) -> Result<(), MetalError> {
        let q = fast_bindings.get(self.q_idx).ok_or(MetalError::InputNotFound("q".into()))?;
        let k = fast_bindings.get(self.k_idx).ok_or(MetalError::InputNotFound("k".into()))?;
        let v = fast_bindings.get(self.v_idx).ok_or(MetalError::InputNotFound("v".into()))?;
        let output = fast_bindings
            .get(self.output_idx)
            .ok_or(MetalError::InputNotFound("output".into()))?;

        let n_heads = self.step.n_heads.resolve(bindings);
        let head_dim = self.step.head_dim.resolve(bindings);
        let kv_seq_len = self.step.kv_seq_len.resolve(bindings);
        // FlashDecodeKernel is a decode-only DSL op: it always dispatches with `q_seq_len=1`.
        // For prefill (M>1), use the main `FlashAttention` op in `metals::sdpa::step`.
        run_flash_decode(foundry, q, k, v, output, n_heads, head_dim, kv_seq_len, 1, self.step.kv_head_major)
    }

    fn name(&self) -> &'static str {
        "FlashDecode"
    }
}
