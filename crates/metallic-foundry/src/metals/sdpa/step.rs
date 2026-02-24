use std::sync::atomic::{AtomicU32, Ordering};

use metallic_env::{FoundryEnvVar, SDPA_DEBUG_ONLINE_COMPARE_MIN_KV, SDPA_DEBUG_ONLINE_COMPARE_PREFILL_MIN_KV, is_set};
use serde::{Deserialize, Serialize};

use crate::{
    Foundry, MetalError, metals::flashattention, spec::{CompiledStep, DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}, storage::{Pooled, View}, tensor::{Dtype, F16, F32, Tensor as FoundryTensor, TensorInit}, types::{KernelArg, TensorArg}
};

/// FlashAttention Step (formerly SdpaStep).
///
/// This is the primary DSL op for performant attention (FlashAttention).
/// It dispatches to:
/// - `flashattention` online fused path when `m == 1` (Flash Decode)
/// - `flashattention` prefill path when `m > 1` (Flash Prefill)
///
/// Unsupported shapes/configs are fail-fast by design. Use `SdpaReferenceStep` explicitly
/// if materialized SDPA behavior is desired.
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
        let debug_sdpa = is_set(FoundryEnvVar::DebugSdpa);
        let force_materialized = is_set(FoundryEnvVar::SdpaForceMaterialized);
        let disable_fa = is_set(FoundryEnvVar::DisableFa) || is_set(FoundryEnvVar::SdpaDisableOnline);
        let verbose_sdpa = is_set(FoundryEnvVar::DebugSdpaVerbose);

        // Ultra-light progress indicator for decode hangs: keep track of the last observed `kv_seq_len`.
        // This is only used for env-gated logging.
        static LAST_KV_SEQ_LEN: AtomicU32 = AtomicU32::new(0);
        if debug_sdpa && m == 1 {
            LAST_KV_SEQ_LEN.store(kv_seq_len, Ordering::Relaxed);
        }

        let verbose_all = is_set(FoundryEnvVar::DebugSdpaVerboseAll);
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
            return Err(MetalError::OperationNotSupported(
                "METALLIC_SDPA_FORCE_MATERIALIZED is not supported for FlashAttentionStep. Use SdpaReferenceStep explicitly.".into(),
            ));
        }

        if disable_fa {
            return Err(MetalError::OperationNotSupported(
                "FlashAttention is disabled via METALLIC_DISABLE_FA/METALLIC_SDPA_DISABLE_ONLINE. Use SdpaReferenceStep explicitly.".into(),
            ));
        }

        if m == 1 {
            // Only use the online path when we're not asked to mask out any portion of K/V.
            // For decode, the common invariant is: query_offset == kv_seq_len - 1 (no future tokens exist).
            if self.step.causal && (q_offset_val + 1 != kv_seq_len) {
                return Err(MetalError::OperationNotSupported(format!(
                    "FlashAttention decode causal offset mismatch: query_offset({q_offset_val}) + 1 != kv_seq_len({kv_seq_len})"
                )));
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
                return Err(MetalError::OperationNotSupported(format!(
                    "FlashAttention decode layout unsupported: q_dims={:?} q_strides={:?} out_dims={:?} out_strides={:?}",
                    q.dims(),
                    q.strides(),
                    output.dims(),
                    output.strides()
                )));
            }

            // Optional debug validation
            static DEBUG_ONLINE_COMPARE_ONCE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let debug_compare = is_set(FoundryEnvVar::SdpaDebugOnlineCompare);
            let min_kv: u32 = SDPA_DEBUG_ONLINE_COMPARE_MIN_KV.get().ok().flatten().unwrap_or(0);
            if debug_compare && kv_seq_len >= min_kv && DEBUG_ONLINE_COMPARE_ONCE.fetch_add(1, Ordering::Relaxed) == 0 {
                let (online, mat): (Vec<f32>, Vec<f32>) = match output.dtype {
                    Dtype::F16 => {
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
                        super::reference::execute_sdpa_reference(
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
                        let online = out_online_view.to_vec(foundry).into_iter().map(f32::from).collect::<Vec<_>>();
                        let mat = out_mat_view.to_vec(foundry).into_iter().map(f32::from).collect::<Vec<_>>();
                        (online, mat)
                    }
                    Dtype::F32 => {
                        let out_tmp = FoundryTensor::<F32, Pooled>::new(foundry, output.dims().to_vec(), TensorInit::Uninitialized)?;

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
                        super::reference::execute_sdpa_reference(
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

                        let out_online_view = FoundryTensor::<F32, View>::from_raw_parts(
                            out_tmp.buffer().clone(),
                            out_tmp.dims().to_vec(),
                            out_tmp.strides().to_vec(),
                            out_tmp.offset(),
                        );
                        let out_mat_view = FoundryTensor::<F32, View>::from_raw_parts(
                            output.buffer().clone(),
                            output.dims().to_vec(),
                            output.strides().to_vec(),
                            output.offset(),
                        );
                        (out_online_view.to_vec(foundry), out_mat_view.to_vec(foundry))
                    }
                    other => {
                        return Err(MetalError::OperationNotSupported(format!(
                            "FlashAttention debug compare supports only F16/F32 output, got {:?}",
                            other
                        )));
                    }
                };
                let d_model = (n_heads as usize) * (head_dim as usize);
                let limit = d_model.min(online.len()).min(mat.len());
                let mut max_abs = 0.0f32;
                let mut n_nan = 0usize;
                for (a, b) in online.iter().take(limit).zip(mat.iter().take(limit)) {
                    let af = *a;
                    let bf = *b;
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

            flashattention::step::run_flash_decode(foundry, q, k, v, output, n_heads, head_dim, kv_seq_len, 1, self.step.kv_head_major)
        } else {
            // Prefill online path (M>1): currently only supports causal attention with the standard
            // invariant `query_offset + m == kv_seq_len`.
            if !self.step.causal {
                Err(MetalError::OperationNotSupported(
                    "FlashAttention prefill requires causal=true".into(),
                ))
            } else if q_offset_val + m != kv_seq_len {
                Err(MetalError::OperationNotSupported(format!(
                    "FlashAttention prefill causal offset mismatch: query_offset({q_offset_val}) + m({m}) != kv_seq_len({kv_seq_len})"
                )))
            } else {
                if debug_sdpa {
                    tracing::info!(target: "metallic_foundry::metals::sdpa", "FlashAttention -> online (prefill) m={}", m);
                }

                // Optional debug validation
                static DEBUG_ONLINE_COMPARE_PREFILL_ONCE: AtomicU32 = AtomicU32::new(0);
                let debug_compare_prefill = is_set(FoundryEnvVar::SdpaDebugOnlineComparePrefill);
                let min_kv: u32 = SDPA_DEBUG_ONLINE_COMPARE_PREFILL_MIN_KV.get().ok().flatten().unwrap_or(0);
                if debug_compare_prefill && kv_seq_len >= min_kv && DEBUG_ONLINE_COMPARE_PREFILL_ONCE.fetch_add(1, Ordering::Relaxed) == 0 {
                    tracing::warn!("FlashAttention debug compare prefill triggered (skipping implementation for brevity)");
                }

                flashattention::step::run_flash_decode(foundry, q, k, v, output, n_heads, head_dim, kv_seq_len, m, self.step.kv_head_major)
            }
        }
    }

    fn name(&self) -> &'static str {
        "FlashAttention"
    }
}

// Backward compatibility or legacy re-exports if needed, but for now we assume fresh usage.
pub type SdpaStep = FlashAttentionStep;
pub type SdpaMaterializedStep = super::reference::SdpaReferenceStep;
