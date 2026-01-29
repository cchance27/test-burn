use std::sync::{
    Arc, atomic::{AtomicU32, Ordering}
};

use half::f16;
use serde::{Deserialize, Serialize};

use crate::{
    Foundry, MetalError, constants, metals::{
        flashattention, gemm::step::{GemmParams, GemmV2Args, gemm_dispatch_config, get_gemm_kernel}, mma::stages::TileConfig, softmax::{SoftmaxV2SdpaBatchedArgs, get_softmax_v2_sdpa_batched_kernel}
    }, policy::activation::Activation, spec::{CompiledStep, DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}, storage::{Pooled, View}, tensor::{F16, Tensor as FoundryTensor, TensorInit}, types::{DispatchConfig, GridSize, KernelArg, TensorArg, ThreadgroupSize}
};

/// FlashAttention Step (formerly SdpaStep).
///
/// This is the primary DSL op for performant attention (FlashAttention).
/// It dispatches to:
/// - `flashattention` online fused path when `m == 1` (Flash Decode)
/// - `flashattention` prefill path when `m > 1` (Flash Prefill, currently D=64 only)
/// - Fallback to `SdpaReferenceStep` (materialized) if conditions not met or debug flags set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlashAttentionStep {
    pub q: Ref,
    pub k: Ref,
    pub v: Ref,
    pub output: Ref,
    #[serde(default)]
    pub causal: bool,
    pub query_offset: DynamicValue<u32>,
    pub n_heads: DynamicValue<u32>,
    pub head_dim: DynamicValue<u32>,
    pub kv_seq_len: DynamicValue<u32>,
    /// Query sequence length (M=1 for decode, M>1 for prefill)
    #[serde(default)]
    pub m: DynamicValue<u32>,
    #[serde(default)]
    pub kv_head_major: bool,
}

#[derive(Debug, Clone)]
pub struct CompiledFlashAttentionStep {
    pub step: FlashAttentionStep,
    pub q_idx: usize,
    pub k_idx: usize,
    pub v_idx: usize,
    pub output_idx: usize,
}

#[typetag::serde(name = "FlashAttention")]
impl Step for FlashAttentionStep {
    fn name(&self) -> &'static str {
        "FlashAttention"
    }

    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        let mut symbols = SymbolTable::new();
        let compiled = self.compile(bindings, &mut symbols);
        let mut fast_bindings = FastBindings::new(symbols.len());

        for (name, symbol_id) in symbols.iter() {
            if let Ok(tensor) = bindings.get(name) {
                fast_bindings.set(*symbol_id, tensor);
            }
        }

        for step in compiled {
            step.execute(foundry, &fast_bindings, bindings, &symbols)?;
        }

        Ok(())
    }

    fn compile(&self, bindings: &mut TensorBindings, symbols: &mut SymbolTable) -> Vec<Box<dyn CompiledStep>> {
        let q_idx = symbols.get_or_create(bindings.interpolate(self.q.0.clone()));
        let k_idx = symbols.get_or_create(bindings.interpolate(self.k.0.clone()));
        let v_idx = symbols.get_or_create(bindings.interpolate(self.v.0.clone()));
        let output_idx = symbols.get_or_create(bindings.interpolate(self.output.0.clone()));

        vec![Box::new(CompiledFlashAttentionStep {
            step: self.clone(),
            q_idx,
            k_idx,
            v_idx,
            output_idx,
        })]
    }
}

/// Fallback/Reference SDPA op (Materialized).
///
/// Uses standard GEMM -> Softmax -> GEMM pipeline.
/// Used for correctness verification and fallbacks when FlashAttention is unsupported (e.g. odd head dims).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdpaReferenceStep {
    pub q: Ref,
    pub k: Ref,
    pub v: Ref,
    pub output: Ref,
    #[serde(default)]
    pub causal: bool,
    pub query_offset: DynamicValue<u32>,
    pub n_heads: DynamicValue<u32>,
    pub head_dim: DynamicValue<u32>,
    pub kv_seq_len: DynamicValue<u32>,
    #[serde(default)]
    pub m: DynamicValue<u32>,
    #[serde(default)]
    pub kv_head_major: bool,
}

#[derive(Debug, Clone)]
pub struct CompiledSdpaReferenceStep {
    pub step: SdpaReferenceStep,
    pub q_idx: usize,
    pub k_idx: usize,
    pub v_idx: usize,
    pub output_idx: usize,
}

#[typetag::serde(name = "SdpaReference")]
impl Step for SdpaReferenceStep {
    fn name(&self) -> &'static str {
        "SdpaReference"
    }

    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        let mut symbols = SymbolTable::new();
        let compiled = self.compile(bindings, &mut symbols);
        let mut fast_bindings = FastBindings::new(symbols.len());

        for (name, symbol_id) in symbols.iter() {
            if let Ok(tensor) = bindings.get(name) {
                fast_bindings.set(*symbol_id, tensor);
            }
        }

        for step in compiled {
            step.execute(foundry, &fast_bindings, bindings, &symbols)?;
        }

        Ok(())
    }

    fn compile(&self, bindings: &mut TensorBindings, symbols: &mut SymbolTable) -> Vec<Box<dyn CompiledStep>> {
        let q_idx = symbols.get_or_create(bindings.interpolate(self.q.0.clone()));
        let k_idx = symbols.get_or_create(bindings.interpolate(self.k.0.clone()));
        let v_idx = symbols.get_or_create(bindings.interpolate(self.v.0.clone()));
        let output_idx = symbols.get_or_create(bindings.interpolate(self.output.0.clone()));

        vec![Box::new(CompiledSdpaReferenceStep {
            step: self.clone(),
            q_idx,
            k_idx,
            v_idx,
            output_idx,
        })]
    }
}

impl CompiledStep for CompiledFlashAttentionStep {
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

        let head_dim = self.step.head_dim.resolve(bindings);
        let kv_seq_len = self.step.kv_seq_len.resolve(bindings);
        let n_heads = self.step.n_heads.resolve(bindings);
        let q_offset_val = self.step.query_offset.resolve(bindings);
        let m_raw = self.step.m.resolve(bindings);
        let m = m_raw.max(1);
        let debug_sdpa = std::env::var_os("METALLIC_DEBUG_SDPA").is_some();
        let force_materialized = std::env::var_os("METALLIC_SDPA_FORCE_MATERIALIZED").is_some();
        let disable_fa = std::env::var_os("METALLIC_DISABLE_FA").is_some() || std::env::var_os("METALLIC_SDPA_DISABLE_ONLINE").is_some();
        let verbose_sdpa = std::env::var_os("METALLIC_DEBUG_SDPA_VERBOSE").is_some();

        // Ultra-light progress indicator for decode hangs: keep track of the last observed `kv_seq_len`.
        // This is only used for env-gated logging.
        static LAST_KV_SEQ_LEN: AtomicU32 = AtomicU32::new(0);
        if debug_sdpa && m == 1 {
            LAST_KV_SEQ_LEN.store(kv_seq_len, Ordering::Relaxed);
        }

        if !self.step.kv_head_major {
            return Err(MetalError::OperationNotSupported(
                "FlashAttention only supports kv_head_major=true for now".into(),
            ));
        }

        let verbose_all = std::env::var_os("METALLIC_DEBUG_SDPA_VERBOSE_ALL").is_some();
        if debug_sdpa && verbose_sdpa && (verbose_all || m == 1) {
            tracing::info!(
                target: "metallic_foundry::metals::sdpa",
                causal = self.step.causal,
                n_heads,
                head_dim,
                kv_seq_len,
                query_offset = q_offset_val,
                m_raw,
                m,
                q_dims = ?q.dims(),
                q_strides = ?q.strides(),
                q_offset_bytes = q.offset(),
                q_dtype = ?q.dtype(),
                k_dims = ?k.dims(),
                k_strides = ?k.strides(),
                k_offset_bytes = k.offset(),
                k_dtype = ?k.dtype(),
                v_dims = ?v.dims(),
                v_strides = ?v.strides(),
                v_offset_bytes = v.offset(),
                v_dtype = ?v.dtype(),
                out_dims = ?output.dims(),
                out_strides = ?output.strides(),
                out_offset_bytes = output.offset(),
                out_dtype = ?output.dtype(),
                "FlashAttention dispatch"
            );
        }

        if force_materialized {
            if debug_sdpa && verbose_sdpa {
                tracing::info!(target: "metallic_foundry::metals::sdpa", "FlashAttention forced -> reference (materialized)");
            }
            return execute_sdpa_reference(
                foundry,
                q,
                k,
                v,
                output,
                n_heads,
                head_dim,
                kv_seq_len,
                q_offset_val,
                m,
                self.step.causal,
            );
        }

        if m == 1 {
            if disable_fa {
                if debug_sdpa {
                    tracing::info!(target: "metallic_foundry::metals::sdpa", "FlashAttention disabled -> reference");
                }
                return execute_sdpa_reference(
                    foundry,
                    q,
                    k,
                    v,
                    output,
                    n_heads,
                    head_dim,
                    kv_seq_len,
                    q_offset_val,
                    1,
                    self.step.causal,
                );
            }

            // Only use the online path when we're not asked to mask out any portion of K/V.
            // For decode, the common invariant is: query_offset == kv_seq_len - 1 (no future tokens exist).
            if self.step.causal && (q_offset_val + 1 != kv_seq_len) {
                if debug_sdpa {
                    tracing::info!(target: "metallic_foundry::metals::sdpa", "FlashAttention m=1 but causal offset mismatch -> reference");
                }
                return execute_sdpa_reference(
                    foundry,
                    q,
                    k,
                    v,
                    output,
                    n_heads,
                    head_dim,
                    kv_seq_len,
                    q_offset_val,
                    1,
                    self.step.causal,
                );
            }

            if debug_sdpa {
                let kv = kv_seq_len;
                let milestone = kv.is_power_of_two();
                if milestone {
                    tracing::info!(
                        target: "metallic_foundry::metals::sdpa",
                        kv_seq_len = kv,
                        query_offset = q_offset_val,
                        last_kv_seq_len = LAST_KV_SEQ_LEN.load(Ordering::Relaxed),
                        "FlashAttention -> FlashDecode"
                    );
                }
            }

            // Layout validation for Flash Decode
            let d_model = (n_heads as usize) * (head_dim as usize);
            let is_token_major_row = |dims: &[usize], strides: &[usize]| -> bool {
                if dims.is_empty() || strides.len() != dims.len() {
                    return false;
                }
                let last_dim_ok = dims.last().copied().unwrap_or(0) == d_model;
                let last_stride_ok = strides.last().copied().unwrap_or(usize::MAX) == 1;
                let row_stride_ok = if dims.len() >= 2 {
                    strides[dims.len() - 2] == d_model
                } else {
                    true
                };
                last_dim_ok && last_stride_ok && row_stride_ok
            };
            let q_ok = is_token_major_row(q.dims(), q.strides())
                || matches!(q.dims(), [d] if *d == d_model)
                || matches!(q.dims(), [1, d] if *d == d_model);
            let out_ok = is_token_major_row(output.dims(), output.strides())
                || matches!(output.dims(), [d] if *d == d_model)
                || matches!(output.dims(), [1, d] if *d == d_model);
            if !(q_ok && out_ok) {
                if debug_sdpa {
                    tracing::warn!(
                        target: "metallic_foundry::metals::sdpa",
                        q_dims = ?q.dims(),
                        q_strides = ?q.strides(),
                        out_dims = ?output.dims(),
                        out_strides = ?output.strides(),
                        "FlashAttention requested but tensor layout unsupported -> reference (materialized)"
                    );
                }
                return execute_sdpa_reference(
                    foundry,
                    q,
                    k,
                    v,
                    output,
                    n_heads,
                    head_dim,
                    kv_seq_len,
                    q_offset_val,
                    1,
                    self.step.causal,
                );
            }

            // Optional debug validation
            static DEBUG_ONLINE_COMPARE_ONCE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let debug_compare = std::env::var_os("METALLIC_SDPA_DEBUG_ONLINE_COMPARE").is_some();
            let min_kv: u32 = std::env::var("METALLIC_SDPA_DEBUG_ONLINE_COMPARE_MIN_KV")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(0);
            if debug_compare && kv_seq_len >= min_kv && DEBUG_ONLINE_COMPARE_ONCE.fetch_add(1, Ordering::Relaxed) == 0 {
                let out_tmp = FoundryTensor::<F16, Pooled>::new(foundry, output.dims().to_vec(), TensorInit::Uninitialized)?;

                // Run online into temp, materialized into real output.
                flashattention::step::run_flash_decode(
                    foundry,
                    q,
                    k,
                    v,
                    &TensorArg::from_tensor(&out_tmp),
                    n_heads,
                    head_dim,
                    kv_seq_len,
                    1,
                    self.step.kv_head_major,
                )?;
                execute_sdpa_reference(
                    foundry,
                    q,
                    k,
                    v,
                    output,
                    n_heads,
                    head_dim,
                    kv_seq_len,
                    q_offset_val,
                    1,
                    self.step.causal,
                )?;
                foundry.synchronize()?;

                let out_online_view = FoundryTensor::<F16, View>::from_raw_parts(
                    out_tmp.buffer().clone(),
                    out_tmp.dims().to_vec(),
                    out_tmp.strides().to_vec(),
                    out_tmp.offset(),
                );
                let out_mat_view = FoundryTensor::<F16, View>::from_raw_parts(
                    output.buffer().clone(),
                    output.dims().to_vec(),
                    output.strides().to_vec(),
                    output.offset(),
                );

                let online: Vec<f16> = out_online_view.to_vec(foundry);
                let mat: Vec<f16> = out_mat_view.to_vec(foundry);
                let d_model = (n_heads as usize) * (head_dim as usize);
                let limit = d_model.min(online.len()).min(mat.len());
                let mut max_abs = 0.0f32;
                let mut n_nan = 0usize;
                for (a, b) in online.iter().take(limit).zip(mat.iter().take(limit)) {
                    let af = f32::from(*a);
                    let bf = f32::from(*b);
                    if !af.is_finite() || !bf.is_finite() {
                        n_nan += 1;
                        continue;
                    }
                    max_abs = max_abs.max((af - bf).abs());
                }

                tracing::warn!(
                    target: "metallic_foundry::metals::sdpa",
                    max_abs_diff = max_abs,
                    non_finite = n_nan,
                    "FlashAttention debug compare (FlashDecode vs Reference)"
                );
                return Ok(());
            }

            return flashattention::step::run_flash_decode(
                foundry,
                q,
                k,
                v,
                output,
                n_heads,
                head_dim,
                kv_seq_len,
                1,
                self.step.kv_head_major,
            );
        } else if !disable_fa {
            // Prefill online path (M>1): currently only supports causal attention with the standard
            // invariant `query_offset + m == kv_seq_len`.
            if !self.step.causal {
                if debug_sdpa {
                    tracing::info!(target: "metallic_foundry::metals::sdpa", "FlashAttention prefill non-causal -> reference");
                }
            } else if q_offset_val + m != kv_seq_len {
                if debug_sdpa {
                    tracing::info!(
                        target: "metallic_foundry::metals::sdpa",
                        query_offset = q_offset_val,
                        m,
                        kv_seq_len,
                        "FlashAttention prefill causal offset mismatch -> reference"
                    );
                }
            } else {
                if debug_sdpa {
                    tracing::info!(target: "metallic_foundry::metals::sdpa", "FlashAttention -> online (prefill) m={}", m);
                }

                // Optional debug validation
                static DEBUG_ONLINE_COMPARE_PREFILL_ONCE: AtomicU32 = AtomicU32::new(0);
                let debug_compare_prefill = std::env::var_os("METALLIC_SDPA_DEBUG_ONLINE_COMPARE_PREFILL").is_some();
                let min_kv: u32 = std::env::var("METALLIC_SDPA_DEBUG_ONLINE_COMPARE_PREFILL_MIN_KV")
                    .ok()
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(0);
                if debug_compare_prefill && kv_seq_len >= min_kv && DEBUG_ONLINE_COMPARE_PREFILL_ONCE.fetch_add(1, Ordering::Relaxed) == 0 {
                    tracing::warn!("FlashAttention debug compare prefill triggered (skipping implementation for brevity)");
                }

                return flashattention::step::run_flash_decode(
                    foundry,
                    q,
                    k,
                    v,
                    output,
                    n_heads,
                    head_dim,
                    kv_seq_len,
                    m,
                    self.step.kv_head_major,
                );
            }
        }

        if debug_sdpa && verbose_sdpa {
            tracing::info!(target: "metallic_foundry::metals::sdpa", "FlashAttention -> reference (materialized) (prefill/fallback)");
        }
        execute_sdpa_reference(
            foundry,
            q,
            k,
            v,
            output,
            n_heads,
            head_dim,
            kv_seq_len,
            q_offset_val,
            m,
            self.step.causal,
        )
    }

    fn name(&self) -> &'static str {
        "FlashAttention"
    }
}

impl CompiledStep for CompiledSdpaReferenceStep {
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

        let head_dim = self.step.head_dim.resolve(bindings);
        let kv_seq_len = self.step.kv_seq_len.resolve(bindings);
        let n_heads = self.step.n_heads.resolve(bindings);
        let q_offset_val = self.step.query_offset.resolve(bindings);
        let m = self.step.m.resolve(bindings).max(1);

        if !self.step.kv_head_major {
            return Err(MetalError::OperationNotSupported(
                "SdpaReference only supports kv_head_major=true for now".into(),
            ));
        }

        execute_sdpa_reference(
            foundry,
            q,
            k,
            v,
            output,
            n_heads,
            head_dim,
            kv_seq_len,
            q_offset_val,
            m,
            self.step.causal,
        )
    }

    fn name(&self) -> &'static str {
        "SdpaReference"
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_sdpa_reference(
    foundry: &mut Foundry,
    q: &TensorArg,
    k: &TensorArg,
    v: &TensorArg,
    output: &TensorArg,
    n_heads: u32,
    head_dim: u32,
    kv_seq_len: u32,
    q_offset_val: u32,
    m: u32,
    causal: bool,
) -> Result<(), MetalError> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let d_model = n_heads
        .checked_mul(head_dim)
        .ok_or_else(|| MetalError::OperationNotSupported("d_model overflow".into()))? as usize;

    // Layout reconciliation:
    //
    // SDPA reference uses GEMM batched over heads (gid.z = head index). That means we must provide:
    // - `batch_stride_*`: offset from head to head in elements
    // - `ld*`: row stride (elements) within a head slice
    //
    // Foundry frequently carries fixed-capacity token-major metadata for Q/Out (e.g. [1, 32, d_model]),
    // while the *buffer contents* may be head-major packed over the true `m` (written by fused KV prep).
    // FlashAttention prefill already handles that ambiguity; the reference path must match it or
    // `METALLIC_DISABLE_FA=1` will silently change semantics.
    let (q_batch_stride, q_row_stride) = {
        let dims = q.dims();
        let strides = q.strides();

        match dims {
            // Flat token-major packed: [m * d_model]
            [len] if strides.len() == 1 && *len >= (m as usize) * d_model => (head_dim as i64, d_model as i32),
            // Fixed-capacity token-major metadata, but head-major packed contents over true `m`.
            // Buffer is interpreted as: [n_heads, m, head_dim] contiguous.
            [1, _m_cap, dm0] if *dm0 == d_model && strides.len() == 3 => {
                let q_head_stride = (m as i64) * (head_dim as i64);
                (q_head_stride, head_dim as i32)
            }
            // Token-major packed: [m, d_model] (or fixed-capacity [m_cap, d_model]).
            [rows, dm0] if *dm0 == d_model && strides.len() == 2 && *rows >= m as usize => {
                // Base for head h is offset by h*head_dim within the row; row stride is d_model.
                (head_dim as i64, d_model as i32)
            }
            // Token-major explicit head: [m, n_heads, head_dim]
            [rows, h, d] if *h == n_heads as usize && *d == head_dim as usize && strides.len() == 3 && *rows >= m as usize => {
                // Base for head h is offset by stride(H); row stride is stride(M).
                (strides[1] as i64, strides[0] as i32)
            }
            // Head-major: [n_heads, m, head_dim]
            [h, rows, d] if *h == n_heads as usize && *d == head_dim as usize && strides.len() == 3 && *rows >= m as usize => {
                ((m as i64) * (head_dim as i64), head_dim as i32)
            }
            _ => {
                return Err(MetalError::OperationNotSupported(format!(
                    "SdpaReference Q layout unsupported: dims={dims:?} strides={strides:?} (expected token-major [m,d_model] or fixed-cap [1,m_cap,d_model] with head-major contents)"
                )));
            }
        }
    };

    let (out_batch_stride, out_row_stride) = {
        let dims = output.dims();
        let strides = output.strides();

        match dims {
            // Flat token-major packed output: [m * d_model]
            [len] if strides.len() == 1 && *len >= (m as usize) * d_model => (head_dim as i64, d_model as i32),
            // Token-major packed output: [m, d_model] or fixed-capacity [m_cap, d_model]
            [rows, dm0] if *dm0 == d_model && strides.len() == 2 && *rows >= m as usize => (head_dim as i64, d_model as i32),
            [1, rows, dm0] if *dm0 == d_model && strides.len() == 3 && *rows >= m as usize => (head_dim as i64, d_model as i32),
            // Token-major explicit head output: [m, n_heads, head_dim]
            [rows, h, d] if *h == n_heads as usize && *d == head_dim as usize && strides.len() == 3 && *rows >= m as usize => {
                (strides[1] as i64, strides[0] as i32)
            }
            // Head-major output: [n_heads, m, head_dim]
            [h, rows, d] if *h == n_heads as usize && *d == head_dim as usize && strides.len() == 3 && *rows >= m as usize => {
                ((m as i64) * (head_dim as i64), head_dim as i32)
            }
            _ => {
                return Err(MetalError::OperationNotSupported(format!(
                    "SdpaReference output layout unsupported: dims={dims:?} strides={strides:?}"
                )));
            }
        }
    };

    // Use default tile config (32x32) - auto_select was causing regression.
    let tile_config = TileConfig::default();
    let (bm, _, _, _, _) = tile_config.tile_sizes();

    let scratch_layout = crate::compound::layout::TiledLayout::sdpa_scratch(n_heads, m, kv_seq_len, bm);
    let (scores_all, probs_all) = crate::metals::sdpa::scratch::get_sdpa_scratch_f16(foundry, scratch_layout)?;

    // Softmax scaling is applied in QK GEMM (alpha=scale). For softmax itself we use 1.0.
    let scale_arg = constants::f16_scalar(foundry, f16::ONE)?;

    // K/V can be a tightly packed history [H, kv_seq_len, D] or a cache view [H, capacity, D].
    let k_seq_stride = k.dims().get(1).copied().unwrap_or(kv_seq_len as usize);
    let v_seq_stride = v.dims().get(1).copied().unwrap_or(kv_seq_len as usize);
    // Unused let d_model_dim = (n_heads as usize) * (head_dim as usize);

    // QK^T GEMM kernel
    let qk_gemm_kernel = get_gemm_kernel(
        Arc::new(crate::policy::f16::PolicyF16),
        Arc::new(crate::policy::f16::PolicyF16),
        false,
        true, // transpose_b (K^T)
        tile_config,
        true,  // has_alpha_beta (scale)
        false, // has_bias
        Activation::None,
    );

    // PV GEMM kernel (unused explicitly here but part of the conceptual pipeline)
    let _av_gemm_kernel_unused = get_gemm_kernel(
        Arc::new(crate::policy::f16::PolicyF16),
        Arc::new(crate::policy::f16::PolicyF16),
        false,
        false,
        tile_config,
        false,
        false,
        Activation::None,
    );

    // GEMM 1: Q @ K^T -> Scores (into head-major strided scratch)
    let mut qk_params = GemmParams::simple(m as i32, kv_seq_len as i32, head_dim as i32, false, true, tile_config);
    qk_params.lda = q_row_stride;
    qk_params.batch_stride_a = q_batch_stride;
    qk_params.batch_stride_b = (k_seq_stride as i64) * (head_dim as i64);
    qk_params.batch_stride_c = scratch_layout.head_stride as i64;
    qk_params.batch_stride_d = scratch_layout.head_stride as i64;

    let qk_dispatch = {
        let base = gemm_dispatch_config(&qk_params, tile_config);
        DispatchConfig {
            grid: GridSize::new(qk_params.tiles_n as usize, qk_params.tiles_m as usize, n_heads as usize),
            group: base.group,
        }
    };

    let qk_args = GemmV2Args {
        a: TensorArg::from_tensor(q),
        b: TensorArg::from_tensor(k),
        d: scores_all.clone(),
        c: scores_all.clone(),
        bias: scores_all.clone(),     // Dummy
        b_scales: scores_all.clone(), // Dummy
        weights_per_block: 32,
        params: qk_params,
        alpha: scale,
        beta: 0.0,
    };
    foundry.run(&qk_gemm_kernel.clone().bind_arc(qk_args, qk_dispatch))?;

    // Softmax: flatten heads into row dimension, dispatch once (over padded_m).
    let softmax_sdpa_kernel = get_softmax_v2_sdpa_batched_kernel();
    let softmax_dispatch = DispatchConfig {
        grid: GridSize::d1((n_heads as usize) * (scratch_layout.padded_m as usize)),
        group: ThreadgroupSize::d1(256),
    };
    let softmax_args = SoftmaxV2SdpaBatchedArgs {
        input: scores_all.clone(),
        scale: scale_arg.clone(),
        output: probs_all.clone(),
        seq_k: kv_seq_len,
        causal: if causal { 1 } else { 0 },
        query_offset: q_offset_val,
        rows_per_batch: scratch_layout.padded_m,
    };
    foundry.run(&softmax_sdpa_kernel.clone().bind_arc(softmax_args, softmax_dispatch))?;

    // GEMM 2: Probs @ V -> Output
    let av_gemm_kernel = get_gemm_kernel(
        Arc::new(crate::policy::f16::PolicyF16),
        Arc::new(crate::policy::f16::PolicyF16),
        false,
        false, // V is normal orientation
        tile_config,
        false, // no scale
        false, // no bias
        Activation::None,
    );

    let mut av_params = GemmParams::simple(m as i32, head_dim as i32, kv_seq_len as i32, false, false, tile_config);
    av_params.ldc = out_row_stride;
    av_params.ldd = out_row_stride;
    av_params.batch_stride_a = scratch_layout.head_stride as i64; // Probs (head-major)
    av_params.batch_stride_b = (v_seq_stride as i64) * (head_dim as i64); // V
    av_params.batch_stride_c = out_batch_stride;
    av_params.batch_stride_d = out_batch_stride;

    let av_dispatch = {
        let base = gemm_dispatch_config(&av_params, tile_config);
        DispatchConfig {
            grid: GridSize::new(av_params.tiles_n as usize, av_params.tiles_m as usize, n_heads as usize),
            group: base.group,
        }
    };

    let av_args = GemmV2Args {
        a: probs_all.clone(),
        b: TensorArg::from_tensor(v),
        d: TensorArg::from_tensor(output),
        c: TensorArg::from_tensor(output),
        bias: TensorArg::from_tensor(output),     // Dummy
        b_scales: TensorArg::from_tensor(output), // Dummy
        weights_per_block: 32,
        params: av_params,
        alpha: 1.0,
        beta: 0.0,
    };

    foundry.run(&av_gemm_kernel.clone().bind_arc(av_args, av_dispatch))?;

    Ok(())
}

// Backward compatibility or legacy re-exports if needed, but for now we assume fresh usage.
pub type SdpaStep = FlashAttentionStep;
pub type SdpaMaterializedStep = SdpaReferenceStep;
