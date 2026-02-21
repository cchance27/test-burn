use std::sync::Arc;

use metallic_macros::KernelArgs;
use serde::{Deserialize, Serialize};

use super::variants::{FlashDecodeVariant, flash_decode_variant_from_env, select_flash_decode_variant_m2m3};
use crate::{
    Foundry, MetalError, compound::{CompiledCompoundKernel, CompoundKernel}, metals::{
        flashattention::stages::{FlashDecodeFusedStage, FlashDecodeStage, HeadLayoutStage, SdpaParams, SdpaPrefillSplitKParams}, rope::{RopeParams, RopeParamsResolved, stage::RopeStage}
    }, spec::{CompiledStep, DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}, storage::Pooled, tensor::{Tensor as FoundryTensor, TensorInit, dtypes::F32}, types::{
        TensorArg, dispatch::{DispatchConfig, GridSize, ThreadgroupSize}
    }
};

/// Get the compiled kernel template.
fn get_rope_flash_decode_kernel(head_dim: u32, variant: FlashDecodeVariant) -> Arc<CompiledCompoundKernel> {
    use crate::kernel_registry::{KernelCacheKey, kernel_registry};
    let suffix = variant.cache_key_suffix();
    let key = KernelCacheKey::new("rope_flash_decode", format!("d{head_dim}_{suffix}"));

    kernel_registry().get_or_build(key, || {
        let name = "rope_flash_decode_v2";
        let dummy_tensor = TensorArg::default();
        let policy = crate::policy::resolve_policy(crate::tensor::Dtype::F16);
        let dummy_layout = HeadLayoutStage::new(
            dummy_tensor.clone(),
            dummy_tensor.clone(),
            dummy_tensor.clone(),
            dummy_tensor.clone(),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            policy.clone(),
        );
        let dummy_rope = RopeStage::new(dummy_tensor.clone(), dummy_tensor.clone(), RopeParams::default());
        let dummy_core: Box<dyn crate::compound::Stage> = match head_dim {
            128 => Box::new(FlashDecodeFusedStage::<128>::new(SdpaParams::default(), variant, policy)),
            64 => Box::new(FlashDecodeFusedStage::<64>::new(SdpaParams::default(), variant, policy)),
            _ => panic!("Unsupported head_dim for dummy core: {}", head_dim),
        };

        CompoundKernel::new(name)
            .prologue(dummy_layout)
            .prologue(dummy_rope)
            .main_dyn(dummy_core)
            .with_manual_output(true)
            .compile()
    })
}

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
        q_strides: (u32, u32),
        k_strides: (u32, u32),
        v_strides: (u32, u32),
        out_strides: (u32, u32),
    ) -> Result<CompiledRopeFlashDecodeStep, MetalError> {
        let variant =
            flash_decode_variant_from_env(head_dim)?.unwrap_or_else(|| select_flash_decode_variant_m2m3(head_dim, sdpa_params.kv_len));
        Self::compile_with_variant(
            _foundry,
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

/// Arguments for the fused RoPE→SDPA compound kernel.
///
/// Buffer binding order must match the stage sequence exactly:
/// `HeadLayoutStage` → `RopeStage` → `FlashDecodeFusedStage*`.
#[derive(KernelArgs)]
struct RopeFlashDecodeArgs {
    // Layout Stage
    q: TensorArg,
    k: TensorArg,
    v: TensorArg,
    output: TensorArg,
    q_stride_b: u32,
    q_stride_h: u32,
    k_stride_b: u32,
    k_stride_h: u32,
    v_stride_b: u32,
    v_stride_h: u32,
    out_stride_b: u32,
    out_stride_h: u32,

    // Rope Stage
    cos: TensorArg,
    sin: TensorArg,
    params_rope: RopeParamsResolved,

    // Core Stage
    sdpa_params: SdpaParams,
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

/// Get or create the standalone Flash Decode kernel (no RoPE fusion).
fn get_flash_decode_kernel(head_dim: u32, variant: FlashDecodeVariant) -> Arc<CompiledCompoundKernel> {
    use crate::kernel_registry::{KernelCacheKey, kernel_registry};

    let suffix = variant.cache_key_suffix();
    let name_suffix = match head_dim {
        128 => format!("d128_{suffix}"),
        64 => format!("d64_{suffix}"),
        _ => panic!("Unsupported head_dim for Flash Decode: {}", head_dim),
    };

    let key = KernelCacheKey::new("flash_decode_standalone", &name_suffix);

    kernel_registry().get_or_build(key, || {
        let policy = crate::policy::resolve_policy(crate::tensor::Dtype::F16);
        let dummy_tensor = TensorArg::default();
        let dummy_layout = HeadLayoutStage::new(
            dummy_tensor.clone(),
            dummy_tensor.clone(),
            dummy_tensor.clone(),
            dummy_tensor.clone(),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            policy.clone(),
        );

        let stage_box: Box<dyn crate::compound::Stage> = match head_dim {
            128 => Box::new(FlashDecodeStage::<128>::new(SdpaParams::default(), variant, policy)),
            64 => Box::new(FlashDecodeStage::<64>::new(SdpaParams::default(), variant, policy)),
            _ => panic!("Unsupported head_dim for Flash Decode: {}", head_dim),
        };

        CompoundKernel::new(&format!("flash_decode_standalone_{}", name_suffix))
            .prologue(dummy_layout)
            .main_dyn(stage_box)
            .with_manual_output(true)
            .compile()
    })
}

/// Arguments for standalone Flash Decode (no RoPE)
#[derive(Debug, KernelArgs)]
struct FlashDecodeArgs {
    q: TensorArg,
    k: TensorArg,
    v: TensorArg,
    #[arg(output)]
    output: TensorArg,
    q_stride_b: u32,
    q_stride_h: u32,
    k_stride_b: u32,
    k_stride_h: u32,
    v_stride_b: u32,
    v_stride_h: u32,
    out_stride_b: u32,
    out_stride_h: u32,
    sdpa_params: SdpaParams,
}

fn parse_env_u32(key: &'static str) -> Option<u32> {
    std::env::var(key).ok().and_then(|s| s.trim().parse::<u32>().ok())
}

#[inline]
fn infer_n_kv_heads(k: &TensorArg, kv_head_major: bool, head_dim: u32) -> u32 {
    if kv_head_major {
        // Head-major:
        // Rank-3: [Heads, Seq, Dim] -> dims[0]
        // Rank-4: [Batch, Heads, Seq, Dim] -> dims[1]
        let dims = k.dims();
        if dims.len() == 3 {
            dims[0] as u32
        } else {
            dims.get(1).copied().unwrap_or(1) as u32
        }
    } else {
        // Token-major: [Batch, Seq, Heads*Dim] (or rank-3 equivalent)
        let last = k.dims().last().copied().unwrap_or(head_dim as usize) as u32;
        (last / head_dim).max(1)
    }
}

/// Get or create the SDPA prefill kernel (tiled, WARPS in {4,8}).
fn get_sdpa_prefill_kernel(prefill_warps: u32) -> Arc<CompiledCompoundKernel> {
    use crate::{
        kernel_registry::{KernelCacheKey, kernel_registry}, metals::flashattention::stages::{SdpaPrefillParams, SdpaPrefillStage, SdpaPrefillVariant}
    };

    let key = KernelCacheKey::new("sdpa_prefill", format!("w{}", prefill_warps));
    kernel_registry().get_or_build(key, || {
        let mut stage = SdpaPrefillStage::new(SdpaPrefillParams::default());
        stage.variant = SdpaPrefillVariant { warps: prefill_warps };
        CompoundKernel::new(&format!("sdpa_prefill_w{}", prefill_warps))
            .main(stage)
            .with_manual_output(true)
            .compile()
    })
}

fn get_sdpa_prefill_splitk_part_kernel(prefill_warps: u32) -> Arc<CompiledCompoundKernel> {
    use crate::{
        kernel_registry::{KernelCacheKey, kernel_registry}, metals::flashattention::stages::{SdpaPrefillSplitKParams, SdpaPrefillSplitKPartStage, SdpaPrefillVariant}
    };

    let key = KernelCacheKey::new("sdpa_prefill_splitk_part", format!("w{}", prefill_warps));
    kernel_registry().get_or_build(key, || {
        let mut stage = SdpaPrefillSplitKPartStage::new(SdpaPrefillSplitKParams::default());
        stage.variant = SdpaPrefillVariant { warps: prefill_warps };
        CompoundKernel::new(&format!("sdpa_prefill_splitk_part_w{}", prefill_warps))
            .main(stage)
            .with_manual_output(true)
            .compile()
    })
}

fn get_sdpa_prefill_splitk_reduce_kernel(prefill_warps: u32) -> Arc<CompiledCompoundKernel> {
    use crate::{
        kernel_registry::{KernelCacheKey, kernel_registry}, metals::flashattention::stages::{SdpaPrefillSplitKParams, SdpaPrefillSplitKReduceStage, SdpaPrefillVariant}
    };

    let key = KernelCacheKey::new("sdpa_prefill_splitk_reduce", format!("w{}", prefill_warps));
    kernel_registry().get_or_build(key, || {
        let mut stage = SdpaPrefillSplitKReduceStage::new(SdpaPrefillSplitKParams::default());
        stage.variant = SdpaPrefillVariant { warps: prefill_warps };
        CompoundKernel::new(&format!("sdpa_prefill_splitk_reduce_w{}", prefill_warps))
            .main(stage)
            .with_manual_output(true)
            .compile()
    })
}

#[derive(Debug, KernelArgs)]
struct SdpaPrefillArgs {
    q: TensorArg,
    k: TensorArg,
    v: TensorArg,
    #[arg(output)]
    output: TensorArg,
    params: crate::metals::flashattention::stages::SdpaPrefillParams,
}

#[derive(Debug, KernelArgs)]
struct SdpaPrefillSplitKPartArgs {
    q: TensorArg,
    k: TensorArg,
    v: TensorArg,
    partial_acc: TensorArg,
    partial_m: TensorArg,
    partial_l: TensorArg,
    params: SdpaPrefillSplitKParams,
}

#[derive(Debug, KernelArgs)]
struct SdpaPrefillSplitKReduceArgs {
    #[arg(output)]
    output: TensorArg,
    partial_acc: TensorArg,
    partial_m: TensorArg,
    partial_l: TensorArg,
    params: SdpaPrefillSplitKParams,
}

#[allow(clippy::too_many_arguments)]
pub fn run_flash_decode_with_variant(
    foundry: &mut Foundry,
    q: &TensorArg,
    k: &TensorArg,
    v: &TensorArg,
    output: &TensorArg,
    n_heads: u32,
    head_dim: u32,
    kv_seq_len: u32,
    q_seq_len: u32,
    kv_head_major: bool,
    variant: FlashDecodeVariant,
) -> Result<(), MetalError> {
    run_flash_decode_impl(
        foundry,
        q,
        k,
        v,
        output,
        n_heads,
        head_dim,
        kv_seq_len,
        q_seq_len,
        kv_head_major,
        Some(variant),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_flash_decode(
    foundry: &mut Foundry,
    q: &TensorArg,
    k: &TensorArg,
    v: &TensorArg,
    output: &TensorArg,
    n_heads: u32,
    head_dim: u32,
    kv_seq_len: u32,
    q_seq_len: u32,
    kv_head_major: bool,
) -> Result<(), MetalError> {
    run_flash_decode_impl(
        foundry,
        q,
        k,
        v,
        output,
        n_heads,
        head_dim,
        kv_seq_len,
        q_seq_len,
        kv_head_major,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_flash_decode_impl(
    foundry: &mut Foundry,
    q: &TensorArg,
    k: &TensorArg,
    v: &TensorArg,
    output: &TensorArg,
    n_heads: u32,
    head_dim: u32,
    kv_seq_len: u32,
    q_seq_len: u32,
    kv_head_major: bool,
    variant_override: Option<FlashDecodeVariant>,
) -> Result<(), MetalError> {
    tracing::trace!(
        n_heads,
        head_dim,
        kv_seq_len,
        q_seq_len,
        kv_head_major,
        has_variant_override = variant_override.is_some(),
        q_dims = ?q.dims(),
        k_dims = ?k.dims(),
        v_dims = ?v.dims(),
        "FlashAttention dispatcher entering"
    );
    // The online SDPA dispatcher handles:
    // - Decode (M=1): head-major Q and head-major KV cache, FlashAttention-style streaming softmax.
    // - Prefill (M>1): tiled kernel for head_dim=64/128.
    if q_seq_len > 1 {
        if variant_override.is_some() {
            return Err(MetalError::OperationNotSupported(
                "run_flash_decode_with_variant is decode-only (q_seq_len must be 1)".into(),
            ));
        }
        // --- PREFILL (M>1) ---

        // Prefill currently supports head_dim==64 and head_dim==128.
        if !(head_dim == 64 || head_dim == 128) {
            return Err(MetalError::OperationNotSupported(format!(
                "Prefill only supports head_dim=64 or 128, got {}",
                head_dim
            )));
        }

        let d_model = n_heads
            .checked_mul(head_dim)
            .ok_or_else(|| MetalError::OperationNotSupported("d_model overflow".into()))?;

        let require_last_contig = |name: &'static str, strides: &[usize]| -> Result<(), MetalError> {
            if strides.last().copied() != Some(1) {
                return Err(MetalError::OperationNotSupported(format!(
                    "Prefill requires contiguous last dim for {name}, got strides={strides:?}"
                )));
            }
            Ok(())
        };

        require_last_contig("q", q.strides())?;
        require_last_contig("k", k.strides())?;
        require_last_contig("v", v.strides())?;
        require_last_contig("output", output.strides())?;

        // Infer Q layout for the prefill kernel.
        //
        // The Metal kernel reads D=64 via lane indices and uses:
        // - per-(head,batch) base pointer: `q + batch*q_stride_b + head*q_stride_h`
        // - per-token step: `q_stride_m` (row-to-row, within a head)
        //
        // NOTE: In Foundry, Q is often allocated as a fixed-capacity token-major tensor for reuse
        // (e.g. `[1, M_cap, d_model]`), but the fused KV-prep writer (`kv_prep_fused`) produces Q
        // as a tightly packed head-major buffer over the *true* `m`. When that happens, the tensor
        // metadata cannot be used to derive `q_stride_h`; it must be `m*head_dim` (not `M_cap*head_dim`).
        let (q_stride_b, q_stride_h, q_stride_m) = {
            let dims = q.dims();
            let strides = q.strides();
            let q_len = q_seq_len as usize;
            let hd = head_dim as usize;
            let dm = d_model as usize;

            // If Q appears token-major (`[1, M_cap, d_model]`) but we're in a head-major KV mode,
            // treat the buffer as head-major packed over `q_seq_len` (the resolved `m`) to match
            // `kv_prep_fused`'s write order.
            let q_head_major_from_token_meta = if kv_head_major {
                if let [1, _m_cap, dm0] = dims {
                    if *dm0 == dm && strides.len() == 3 {
                        let q_stride_h = q_seq_len
                            .checked_mul(head_dim)
                            .ok_or_else(|| MetalError::OperationNotSupported("Prefill Q stride overflow".into()))?;
                        let q_stride_b = n_heads
                            .checked_mul(q_stride_h)
                            .ok_or_else(|| MetalError::OperationNotSupported("Prefill Q stride overflow".into()))?;
                        Some((q_stride_b, q_stride_h, head_dim))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let strides = if let Some(strides) = q_head_major_from_token_meta {
                strides
            } else {
                match dims {
                    // Head-major: [H, M, D]
                    [h, m, d] if *h == n_heads as usize && *m >= q_len && *d == hd && strides.len() == 3 => {
                        (0u32, strides[0] as u32, strides[1] as u32)
                    }
                    // Head-major w/ explicit batch: [B, H, M, D]
                    [b, h, m, d] if *h == n_heads as usize && *m >= q_len && *d == hd && strides.len() == 4 => {
                        (strides[0] as u32, strides[1] as u32, strides[2] as u32)
                    }
                    // Token-major packed: [M, DModel]
                    [m, dm0] if *m >= q_len && *dm0 == dm && strides.len() == 2 => (0u32, head_dim, strides[0] as u32),
                    // Token-major packed w/ batch: [B, M, DModel]
                    [b, m, dm0] if *m >= q_len && *dm0 == dm && strides.len() == 3 => (strides[0] as u32, head_dim, strides[1] as u32),
                    // Token-major explicit head: [M, H, D]
                    [m, h, d] if *m >= q_len && *h == n_heads as usize && *d == hd && strides.len() == 3 => {
                        (0u32, strides[1] as u32, strides[0] as u32)
                    }
                    // Token-major explicit head w/ batch: [B, M, H, D]
                    [b, m, h, d] if *m >= q_len && *h == n_heads as usize && *d == hd && strides.len() == 4 => {
                        (strides[0] as u32, strides[2] as u32, strides[1] as u32)
                    }
                    _ => {
                        return Err(MetalError::OperationNotSupported(format!(
                            "Prefill Q layout unsupported: dims={dims:?} strides={strides:?} (expected head-major [H,M>=q_len,D] or token-major [M>=q_len,d_model]/[M>=q_len,H,D])"
                        )));
                    }
                }
            };
            Ok::<(u32, u32, u32), MetalError>(strides)
        }?;

        let (out_stride_b, out_stride_h, out_stride_m) = {
            let dims = output.dims();
            let strides = output.strides();
            let q_len = q_seq_len as usize;
            let hd = head_dim as usize;
            let dm = d_model as usize;

            match dims {
                // Packed token-major: [M, DModel]
                [m, dm0] if *m >= q_len && *dm0 == dm && strides.len() == 2 => (0u32, head_dim, strides[0] as u32),
                // Packed token-major w/ batch: [B, M, DModel]
                [b, m, dm0] if *m >= q_len && *dm0 == dm && strides.len() == 3 => (strides[0] as u32, head_dim, strides[1] as u32),
                // Token-major explicit head: [M, H, D]
                [m, h, d] if *m >= q_len && *h == n_heads as usize && *d == hd && strides.len() == 3 => {
                    (0u32, strides[1] as u32, strides[0] as u32)
                }
                // Token-major explicit head w/ batch: [B, M, H, D]
                [b, m, h, d] if *m >= q_len && *h == n_heads as usize && *d == hd && strides.len() == 4 => {
                    (strides[0] as u32, strides[2] as u32, strides[1] as u32)
                }
                // Head-major output: [H, M, D]
                [h, m, d] if *h == n_heads as usize && *m >= q_len && *d == hd && strides.len() == 3 => {
                    (0u32, strides[0] as u32, strides[1] as u32)
                }
                // Head-major w/ batch: [B, H, M, D]
                [b, h, m, d] if *h == n_heads as usize && *m >= q_len && *d == hd && strides.len() == 4 => {
                    (strides[0] as u32, strides[1] as u32, strides[2] as u32)
                }
                _ => {
                    return Err(MetalError::OperationNotSupported(format!(
                        "Prefill output layout unsupported: dims={dims:?} strides={strides:?}"
                    )));
                }
            }
        };

        let n_kv_heads = infer_n_kv_heads(k, kv_head_major, head_dim);

        let mut group_size = n_heads / n_kv_heads;
        if group_size == 0 {
            group_size = 1;
        }
        let kv_path = if kv_head_major && n_kv_heads < n_heads {
            "compact_gqa"
        } else if kv_head_major {
            "expanded_heads_or_mha"
        } else {
            "token_major"
        };
        tracing::trace!(
            kv_path,
            n_heads,
            n_kv_heads,
            group_size,
            kv_head_major,
            q_seq_len,
            kv_seq_len,
            q_dims = ?q.dims(),
            k_dims = ?k.dims(),
            v_dims = ?v.dims(),
            "FlashAttention prefill KV path resolved"
        );

        if std::env::var("METALLIC_DEBUG_SDPA").is_ok() {
            tracing::info!("SDPA Prefill Debug: Q dims={:?} K dims={:?}", q.dims(), k.dims());
            tracing::info!(
                "SDPA Prefill Debug: n_heads={} n_kv_heads={} group_size={} kv_head_major={}",
                n_heads,
                n_kv_heads,
                group_size,
                kv_head_major
            );
            tracing::info!("SDPA Prefill Debug: K strides={:?}", k.strides());
        }

        let (k_stride_b, k_stride_h, stride_k_s) = {
            let s = k.strides();
            if kv_head_major {
                if s.len() == 4 {
                    (s[0] as u32, s[1] as u32, s[2] as u32)
                } else {
                    // Rank-3 head-major: [H, Seq, D].
                    // Prefill currently dispatches with batch=1, so treat batch stride as 0.
                    (0, s[0] as u32, s[1] as u32)
                }
            } else {
                // Token-Major [B, M, H, D]
                if s.len() == 4 {
                    (s[0] as u32, s[2] as u32, s[1] as u32)
                } else {
                    // Rank 3 [M, H, D]
                    (0, s[1] as u32, s[0] as u32)
                }
            }
        };

        let (v_stride_b, v_stride_h, stride_v_s) = {
            let s = v.strides();
            if kv_head_major {
                if s.len() == 4 {
                    (s[0] as u32, s[1] as u32, s[2] as u32)
                } else {
                    (0, s[0] as u32, s[1] as u32)
                }
            } else if s.len() == 4 {
                (s[0] as u32, s[2] as u32, s[1] as u32)
            } else {
                (0, s[1] as u32, s[0] as u32)
            }
        };

        let prefill_warps = parse_env_u32("METALLIC_FA_PREFILL_WARPS").unwrap_or(8);
        if !matches!(prefill_warps, 4 | 8) {
            return Err(MetalError::OperationNotSupported(format!(
                "METALLIC_FA_PREFILL_WARPS must be 4 or 8, got {}",
                prefill_warps
            )));
        }

        let tiling_m = prefill_warps * 4;
        let grid_m_tiles = (q_seq_len + tiling_m - 1) / tiling_m;

        // Split-K selector (FA1 completion for large KV prefill).
        //
        // Env overrides:
        // - METALLIC_DISABLE_FA_PREFILL_SPLITK=1 -> force split_k=1 (disabled)
        // - METALLIC_FA_PREFILL_SPLIT_K=N        -> force split_k=N (N<=1 disables)
        let mut split_k = if std::env::var("METALLIC_DISABLE_FA_PREFILL_SPLITK").is_ok() {
            1u32
        } else if let Some(v) = parse_env_u32("METALLIC_FA_PREFILL_SPLIT_K") {
            v.max(1)
        } else {
            // Conservative default heuristic; tune via sweep.
            if kv_seq_len >= 4096 && q_seq_len >= 16 {
                8
            } else if kv_seq_len >= 2048 && q_seq_len >= 16 {
                4
            } else {
                1
            }
        };
        let kv_tiles = (kv_seq_len + 31) / 32;
        split_k = split_k.min(kv_tiles.max(1));

        // In Foundry compound kernels, `gid` is `[[threadgroup_position_in_grid]]`, so `grid.x` is
        // the number of *threadgroups* (tiles), not the number of threads.
        let group = ThreadgroupSize::d1((prefill_warps * 32) as usize);

        let scale = 1.0 / (head_dim as f32).sqrt();
        // Prefill uses the common causal invariant: `query_offset + q_len == kv_len`.
        // The prefill kernel needs `query_offset` so that causal masking compares each query row
        // against its absolute position in the KV stream.
        let query_offset = kv_seq_len.saturating_sub(q_seq_len);

        use crate::metals::flashattention::stages::SdpaPrefillParams;
        let args = SdpaPrefillArgs {
            q: TensorArg::from_tensor(q),
            k: TensorArg::from_tensor(k),
            v: TensorArg::from_tensor(v),
            output: TensorArg::from_tensor(output),
            params: SdpaPrefillParams {
                kv_len: kv_seq_len,
                head_dim,
                scale,
                stride_k_s,
                stride_v_s,
                query_offset,
                q_stride_b,
                q_stride_h,
                k_stride_b,
                k_stride_h,
                v_stride_b,
                v_stride_h,
                out_stride_b,
                out_stride_h,

                q_stride_m,
                out_stride_m,

                group_size,

                q_len: q_seq_len,
            },
        };

        if split_k <= 1 {
            let grid = GridSize::new(grid_m_tiles as usize, n_heads as usize, 1 /* batch=1 assumed */);
            let config = DispatchConfig::new(grid, group);
            let kernel = get_sdpa_prefill_kernel(prefill_warps);
            let bound = kernel.clone().bind_arc(args, config);
            return foundry.run(&bound);
        }

        // Split-K path: 2-phase (part + reduce) using FP32 scratch.
        let q_tile_count = grid_m_tiles;
        let tile_m = tiling_m;
        let partial_rows = (split_k as usize)
            .checked_mul(n_heads as usize)
            .and_then(|v| v.checked_mul(q_tile_count as usize))
            .and_then(|v| v.checked_mul(tile_m as usize))
            .ok_or_else(|| MetalError::OperationNotSupported("Split-K scratch size overflow".into()))?;
        let partial_acc_elems = partial_rows
            .checked_mul(head_dim as usize)
            .ok_or_else(|| MetalError::OperationNotSupported("Split-K acc scratch size overflow".into()))?;

        let partial_acc = FoundryTensor::<F32, Pooled>::new(foundry, vec![partial_acc_elems], TensorInit::Uninitialized)?;
        let partial_m = FoundryTensor::<F32, Pooled>::new(foundry, vec![partial_rows], TensorInit::Uninitialized)?;
        let partial_l = FoundryTensor::<F32, Pooled>::new(foundry, vec![partial_rows], TensorInit::Uninitialized)?;

        let splitk_params = SdpaPrefillSplitKParams {
            kv_len: kv_seq_len,
            head_dim,
            scale,
            stride_k_s,
            stride_v_s,
            query_offset,
            q_stride_b,
            q_stride_h,
            k_stride_b,
            k_stride_h,
            v_stride_b,
            v_stride_h,
            out_stride_b,
            out_stride_h,
            q_stride_m,
            out_stride_m,
            group_size,
            q_len: q_seq_len,
            n_heads,
            split_k,
        };

        let part_args = SdpaPrefillSplitKPartArgs {
            q: TensorArg::from_tensor(q),
            k: TensorArg::from_tensor(k),
            v: TensorArg::from_tensor(v),
            partial_acc: TensorArg::from_tensor(&partial_acc),
            partial_m: TensorArg::from_tensor(&partial_m),
            partial_l: TensorArg::from_tensor(&partial_l),
            params: splitk_params,
        };
        let part_grid = GridSize::new(grid_m_tiles as usize, n_heads as usize, split_k as usize);
        let part_config = DispatchConfig::new(part_grid, group);
        let part_kernel = get_sdpa_prefill_splitk_part_kernel(prefill_warps);
        let part_bound = part_kernel.clone().bind_arc(part_args, part_config);
        foundry.run(&part_bound)?;

        let reduce_args = SdpaPrefillSplitKReduceArgs {
            output: TensorArg::from_tensor(output),
            partial_acc: TensorArg::from_tensor(&partial_acc),
            partial_m: TensorArg::from_tensor(&partial_m),
            partial_l: TensorArg::from_tensor(&partial_l),
            params: splitk_params,
        };
        let reduce_grid = GridSize::new(grid_m_tiles as usize, n_heads as usize, 1);
        let reduce_config = DispatchConfig::new(reduce_grid, group);
        let reduce_kernel = get_sdpa_prefill_splitk_reduce_kernel(prefill_warps);
        let reduce_bound = reduce_kernel.clone().bind_arc(reduce_args, reduce_config);
        return foundry.run(&reduce_bound);
    }

    // --- DECODE PATH (M=1) ---

    // Unsupported head dim check for standalone decode
    if head_dim != 64 && head_dim != 128 {
        return Err(MetalError::OperationNotSupported(format!(
            "Flash Decode only supports head_dim=64 or 128, got {}",
            head_dim
        )));
    }

    // Compute strides based on head-major cache layout [cache_heads, allocated_capacity, head_dim].
    let batch = 1u32;
    let capacity = k.dims().get(1).copied().unwrap_or(kv_seq_len as usize) as u32;

    // Q is produced by KvPrepFused (and historical KvRearrange+RoPE) in head-major layout:
    // [n_heads, q_seq_len, head_dim], flattened contiguous.
    //
    // Note: many Foundry tensors keep token-major dims/strides metadata (e.g. [1, 32, d_model]),
    // but the buffer contents for q_rot are still head-major. We must therefore use the
    // resolved q_seq_len/head_dim to compute offsets, not TensorArg's strides.
    let q_seq_len_decode = q_seq_len.max(1);
    let (q_stride_b, q_stride_h) = if kv_head_major {
        (n_heads * q_seq_len_decode * head_dim, q_seq_len_decode * head_dim)
    } else {
        // Token-Major: [B, M, H, D]
        // Stride B: M * H * D
        // Stride H: D
        (q_seq_len_decode * n_heads * head_dim, head_dim)
    };
    let n_kv_heads = infer_n_kv_heads(k, kv_head_major, head_dim).max(1);
    let cache_heads = if kv_head_major { n_kv_heads } else { n_heads };
    let group_size = (n_heads / n_kv_heads).max(1);
    let kv_path = if kv_head_major && n_kv_heads < n_heads {
        "compact_gqa"
    } else if kv_head_major {
        "expanded_heads_or_mha"
    } else {
        "token_major"
    };
    tracing::trace!(
        kv_path,
        n_heads,
        n_kv_heads,
        cache_heads,
        group_size,
        kv_head_major,
        capacity,
        kv_seq_len,
        q_dims = ?q.dims(),
        k_dims = ?k.dims(),
        v_dims = ?v.dims(),
        "FlashAttention decode KV path resolved"
    );

    // Use actual cache head count for head stride to support both expanded and compact GQA caches.
    let (k_stride_b, k_stride_h) = (cache_heads * capacity * head_dim, capacity * head_dim);
    let (v_stride_b, v_stride_h) = (cache_heads * capacity * head_dim, capacity * head_dim);
    let (out_stride_b, out_stride_h) = (n_heads * head_dim, head_dim);

    let scale = 1.0 / (head_dim as f32).sqrt();

    let sdpa_params = SdpaParams {
        kv_len: kv_seq_len,
        head_dim,
        scale,
        stride_k_s: head_dim,
        stride_v_s: head_dim,
    };

    let args = FlashDecodeArgs {
        q: TensorArg::from_tensor(q),
        k: TensorArg::from_tensor(k),
        v: TensorArg::from_tensor(v),
        output: TensorArg::from_tensor(output),
        q_stride_b,
        q_stride_h,
        k_stride_b,
        k_stride_h,
        v_stride_b,
        v_stride_h,
        out_stride_b,
        out_stride_h,
        sdpa_params,
    };

    let variant = if let Some(v) = variant_override {
        v
    } else if let Some(v) = flash_decode_variant_from_env(head_dim)? {
        v
    } else {
        select_flash_decode_variant_m2m3(head_dim, kv_seq_len)
    };
    variant.validate_for_head_dim(head_dim)?;

    // Dispatch: Grid (1, Heads, Batch) ⇒ one threadgroup per (head, batch).
    let grid = GridSize::new(1, n_heads as usize, batch as usize);
    let group = ThreadgroupSize::d1(variant.threads_per_tg() as usize);
    let config = DispatchConfig::new(grid, group);

    let kernel = get_flash_decode_kernel(head_dim, variant);

    let bound = kernel.clone().bind_arc(args, config);
    foundry.run(&bound)
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
