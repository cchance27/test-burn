use std::sync::OnceLock;

use metallic_instrumentation::{MetricEvent, record_metric_async};

use super::{main::Context, utils::MatMulBackendOverride as InternalMatMulBackendOverride};
use crate::{
    MetalError, Tensor, caching::ResourceCache, kernels::{
        elemwise_add::BroadcastElemwiseAddInplaceOp, matmul_gemv::{
            MATMUL_Q8_NT_MAX_ROWS, MatmulF16CanonicalOp, MatmulF16CanonicalQkvFusedOp, MatmulF16CanonicalRows16Op, MatmulF16CanonicalSwiGluOp, MatmulGemvAddmmOp, MatmulGemvOp, MatmulGemvSmallMOp, MatmulQ8CanonicalOp, MatmulQ8CanonicalRows16Op, MatmulQ8NtOp
        }, matmul_gemv_qkv_fused::MatmulGemvQkvFusedOp, matmul_mlx::MatMulMlxOp, matmul_mps::{MatMulBackend, MatMulMpsAlphaBetaOp, MatMulMpsOp}
    }, tensor::{CanonicalF16Tensor, QuantizedTensor, TensorElement, TensorType}
};

const METALLIC_Q8_M1_MLX_MIN_N_ENV: &str = "METALLIC_Q8_M1_MLX_MIN_N";
const METALLIC_F16_CANONICAL_GEMM_ENV: &str = "METALLIC_F16_CANONICAL_GEMM";

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(test)]
static Q8_M1_MLX_MIN_N_OVERRIDE: AtomicUsize = AtomicUsize::new(usize::MAX);

#[inline]
fn q8_m1_mlx_min_n() -> usize {
    #[cfg(test)]
    {
        let v = Q8_M1_MLX_MIN_N_OVERRIDE.load(Ordering::Relaxed);
        if v != usize::MAX {
            return v;
        }
    }

    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var(METALLIC_Q8_M1_MLX_MIN_N_ENV)
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            // Default: only prefer MLX for very large `n` (e.g. FFN up/gate at 4864, logits at vocab).
            .unwrap_or(4096)
    })
}

#[cfg(test)]
pub(crate) fn set_q8_m1_mlx_min_n_override_for_tests(value: Option<usize>) {
    Q8_M1_MLX_MIN_N_OVERRIDE.store(value.unwrap_or(usize::MAX), Ordering::Relaxed);
}

#[inline]
pub(crate) fn f16_canonical_gemm_enabled() -> bool {
    std::env::var(METALLIC_F16_CANONICAL_GEMM_ENV)
        .ok()
        .map(|s| s.trim() != "0")
        .unwrap_or(true)
}

#[inline]
pub(crate) fn q8_should_use_mlx_for_m1(dims: &MatmulDims, transpose_a: bool, transpose_b: bool) -> bool {
    dims.batch == 1 && dims.m == 1 && !transpose_a && !transpose_b && dims.n >= q8_m1_mlx_min_n()
}

#[inline]
fn matmul_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("METALLIC_TRACE_MATMUL_DISPATCH")
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(false)
    })
}

#[inline]
fn emit_matmul_backend_selected(
    rhs: &'static str,
    dims: Option<MatmulDims>,
    transpose_a: bool,
    transpose_b: bool,
    backend: &'static str,
    reason: impl Into<String>,
) {
    if !matmul_trace_enabled() {
        return;
    }

    let op_name = if let Some(d) = dims {
        format!(
            "matmul_{rhs}(m={m},n={n},k={k},ta={ta},tb={tb},b={batch})",
            rhs = rhs,
            m = d.m,
            n = d.n,
            k = d.k,
            ta = transpose_a as u8,
            tb = transpose_b as u8,
            batch = d.batch
        )
    } else {
        format!(
            "matmul_{rhs}(ta={ta},tb={tb})",
            rhs = rhs,
            ta = transpose_a as u8,
            tb = transpose_b as u8
        )
    };

    record_metric_async!(MetricEvent::KernelBackendSelected {
        op_name,
        backend: backend.to_string(),
        reason: reason.into(),
    });
}

#[derive(Debug, Clone, Copy)]
pub struct MatmulDims {
    pub batch: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

#[derive(Clone, Copy)]
pub struct MatmulAlphaBeta<'a, T: TensorElement> {
    pub output: &'a Tensor<T>,
    pub alpha: f32,
    pub beta: f32,
}

pub enum QkvWeights<'a, T: TensorElement> {
    Dense {
        fused_weight: &'a Tensor<T>,
        fused_bias: &'a Tensor<T>,
        d_model: usize,
        kv_dim: usize,
    },
    Quantized {
        wq: &'a crate::tensor::QuantizedQ8_0Tensor,
        wk: &'a crate::tensor::QuantizedQ8_0Tensor,
        wv: &'a crate::tensor::QuantizedQ8_0Tensor,
        q_bias: &'a Tensor<T>,
        k_bias: &'a Tensor<T>,
        v_bias: &'a Tensor<T>,
    },
    DenseCanonical {
        wq: &'a CanonicalF16Tensor<T>,
        wk: &'a CanonicalF16Tensor<T>,
        wv: &'a CanonicalF16Tensor<T>,
        q_bias: &'a Tensor<T>,
        k_bias: &'a Tensor<T>,
        v_bias: &'a Tensor<T>,
    },
}

impl<T: TensorElement> Context<T> {
    #[inline]
    fn smallm_fallback_max_n() -> usize {
        std::env::var("METALLIC_Q8_SMALLM_FALLBACK_MAX_N")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4096)
    }

    #[inline]
    fn canonical_force_enabled() -> bool {
        std::env::var("METALLIC_Q8_CANONICAL_N").ok().map(|s| s != "0").unwrap_or(false)
    }

    #[inline]
    fn canonical_min_n_threshold() -> usize {
        std::env::var("METALLIC_Q8_CANONICAL_MIN_N")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4096)
    }

    #[inline]
    fn canonical_max_m() -> usize {
        std::env::var("METALLIC_Q8_CANONICAL_MAX_M")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(16)
    }

    #[inline]
    fn should_use_q8_canonical(&self, dims: &MatmulDims, transpose_a: bool) -> bool {
        if transpose_a || dims.batch != 1 {
            return false;
        }
        if Self::canonical_force_enabled() {
            return true;
        }
        if dims.m < 2 || dims.n == 0 {
            return false;
        }
        if dims.m > Self::canonical_max_m() {
            return false;
        }
        dims.n >= Self::canonical_min_n_threshold()
    }

    #[inline]
    fn canonical_rows16_cutoff_n(m: usize) -> usize {
        // Allow override, else use m-specific crossover observed in benches
        if let Ok(s) = std::env::var("METALLIC_Q8_CANONICAL_R16_MAX_N")
            && let Ok(val) = s.parse::<usize>()
        {
            return val;
        }
        match m {
            2..=4 => 1536,
            _ => 0,
        }
    }

    #[inline]
    fn should_use_q8_canonical_rows16(&self, dims: &MatmulDims, transpose_a: bool) -> bool {
        if transpose_a || dims.batch != 1 {
            return false;
        }
        if dims.m < 2 || dims.n < 512 {
            return false;
        }
        if dims.m > 4 {
            return false;
        }
        let cutoff = Self::canonical_rows16_cutoff_n(dims.m);
        if cutoff == 0 {
            return false;
        }
        dims.n <= cutoff
    }

    #[inline]
    fn f16_canonical_rows16_cutoff_n(m: usize) -> usize {
        if let Ok(s) = std::env::var("METALLIC_F16_CANONICAL_R16_MAX_N")
            && let Ok(val) = s.parse::<usize>()
        {
            return val;
        }
        match m {
            2..=4 => 0,
            _ => 0,
        }
    }

    #[inline]
    fn should_use_f16_canonical_rows16(&self, dims: &MatmulDims, transpose_a: bool, transpose_b: bool) -> bool {
        if transpose_a || transpose_b || dims.batch != 1 {
            return false;
        }
        if dims.m < 2 || dims.m > 4 {
            return false;
        }
        let cutoff = Self::f16_canonical_rows16_cutoff_n(dims.m);
        if cutoff == 0 {
            return false;
        }
        dims.n <= cutoff
    }

    // Device capability hints used by dispatcher heuristics. These are conservative defaults
    // and should be updated to query real device/feature sets.
    #[inline]
    pub fn device_has_simdgroup_mm(&self) -> bool {
        // TODO(DEBT): Detect simdgroup matrix multiply support via Metal feature sets / GPU family.
        // For now, return false; dispatcher thresholds can be tuned via env.
        false
    }

    #[inline]
    pub fn max_threads_per_threadgroup(&self) -> usize {
        // TODO(DEBT): Provide a pipeline-aware value; fall back to a conservative upper bound.
        1024
    }

    // Compute matmul dims when RHS can be dense or quantized.
    fn compute_matmul_dims(
        &self,
        a: &Tensor<T>,
        b: &TensorType<T>,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<MatmulDims, MetalError> {
        match b {
            TensorType::Dense(bd) => {
                // self.compute_matmul_dims(a, bd, transpose_a, transpose_b),
                let a_view = a.as_mps_matrix_batch_view()?;
                let b_view = bd.as_mps_matrix_batch_view()?;
                let (a_rows, a_cols) = if transpose_a {
                    (a_view.columns, a_view.rows)
                } else {
                    (a_view.rows, a_view.columns)
                };
                let (b_rows, b_cols) = if transpose_b {
                    (b_view.columns, b_view.rows)
                } else {
                    (b_view.rows, b_view.columns)
                };
                if a_cols != b_rows {
                    return Err(MetalError::InvalidOperation("matmul dims mismatch".into()));
                }
                Ok(MatmulDims {
                    batch: a_view.batch.max(b_view.batch),
                    m: a_rows,
                    n: b_cols,
                    k: a_cols,
                })
            }
            TensorType::DenseCanonical(bc) => {
                if transpose_b {
                    return Err(MetalError::OperationNotSupported(
                        "canonical matmul does not support transpose_b".into(),
                    ));
                }
                let a_view = a.as_mps_matrix_batch_view()?;
                let (a_rows, a_cols) = if transpose_a {
                    (a_view.columns, a_view.rows)
                } else {
                    (a_view.rows, a_view.columns)
                };
                if bc.logical_dims.len() != 2 {
                    return Err(MetalError::InvalidShape("Canonical RHS must be 2D".into()));
                }
                let b_rows = bc.logical_dims[0];
                let b_cols = bc.logical_dims[1];
                if a_cols != b_rows {
                    return Err(MetalError::InvalidOperation("matmul dims mismatch (canonical)".into()));
                }
                Ok(MatmulDims {
                    batch: a_view.batch,
                    m: a_rows,
                    n: b_cols,
                    k: a_cols,
                })
            }
            TensorType::Quant(qrhs) => {
                // a is dense; use its view to get (batch, rows, cols)
                let a_view = a.as_mps_matrix_batch_view()?;
                let (a_rows, a_cols) = if transpose_a {
                    (a_view.columns, a_view.rows)
                } else {
                    (a_view.rows, a_view.columns)
                };
                // derive b rows/cols from quant dims
                let (b_rows, b_cols) = match qrhs {
                    QuantizedTensor::Q8_0(q8) => {
                        if q8.logical_dims.len() != 2 {
                            return Err(MetalError::InvalidShape("Quant RHS must be 2D".into()));
                        }
                        let d0 = q8.logical_dims[0];
                        let d1 = q8.logical_dims[1];
                        // Determine which dim is K by matching a_cols.
                        let dims_kn = if d0 == a_cols {
                            true
                        } else if d1 == a_cols {
                            false
                        } else {
                            return Err(MetalError::InvalidShape("Quant RHS dims mismatch with A".into()));
                        };
                        // Quant logical dims correspond to the underlying (K, N) matrix regardless of transpose flag.
                        // Our quant dispatch selects the appropriate kernel (GEMV or GEMM_NT) using `transpose_b`,
                        // so here we always report dims as (K, N) so that `a_cols == b_rows` holds.
                        let (k_dim, n_dim) = if dims_kn { (d0, d1) } else { (d1, d0) };
                        (k_dim, n_dim)
                    }
                };
                if a_cols != b_rows {
                    return Err(MetalError::InvalidOperation("matmul dims mismatch (quant)".into()));
                }
                Ok(MatmulDims {
                    batch: a_view.batch,
                    m: a_rows,
                    n: b_cols,
                    k: a_cols,
                })
            }
        }
    }

    #[inline]
    fn should_use_mlx_bias(&self, dims: &MatmulDims) -> bool {
        if dims.n <= 32 {
            return false;
        }

        if dims.batch == 1 && dims.m <= 4 {
            // For very skinny decode projections benchmark data shows MLX holds an advantage
            // unless both the output width is extremely small and the reduction dim dwarfs it.
            if dims.n <= 1024 && dims.k >= dims.n {
                return false;
            }
        }

        if dims.m >= 1024 || dims.n >= 1024 {
            return true;
        }

        if dims.batch > 1 && dims.m >= 256 && dims.n >= 256 {
            return true;
        }

        false
    }

    #[inline]
    fn has_strided_mps_batch(&self, tensors: &[&Tensor<T>]) -> bool {
        tensors.iter().any(|tensor| {
            tensor
                .as_mps_matrix_batch_view()
                .map(|view| view.batch > 1 && view.matrix_bytes != view.rows * view.row_bytes)
                .unwrap_or(false)
        })
    }

    #[inline]
    fn should_use_mlx_dense(&self, dims: &MatmulDims, has_strided_batch: bool) -> bool {
        if dims.n <= 32 && !has_strided_batch {
            return false;
        }

        if dims.batch == 1 && dims.m <= 4 {
            if !has_strided_batch && dims.n <= 128 && dims.k >= dims.n * 2 {
                return false;
            }
            if !has_strided_batch {
                let four_k = dims.k.saturating_mul(4);
                let four_n = dims.n.saturating_mul(4);
                if dims.n >= four_k || dims.k >= four_n {
                    return false;
                }
            }
        }

        if dims.m >= 1024 || dims.n >= 1024 {
            return true;
        }

        true
    }

    #[inline]
    fn can_use_gemv(&self, dims: &MatmulDims, transpose_a: bool, transpose_b: bool) -> bool {
        // SIMD GEMV kernels (MatmulGemvCols8) only support F16, not F32.
        if T::DTYPE != crate::tensor::Dtype::F16 {
            return false;
        }
        if transpose_a {
            return false;
        }
        // Allow transpose_b (standard linear layer) because our kernel supports [N, K] weights
        // which matches the execution semantic of `x @ W^T` with contiguous K.
        if !transpose_b {
            return false;
        }

        if dims.batch != 1 {
            return false;
        }

        dims.m == 1
    }

    #[inline]
    fn can_use_gemv_canonical(&self, dims: &MatmulDims, transpose_a: bool, transpose_b: bool) -> bool {
        if T::DTYPE != crate::tensor::Dtype::F16 {
            return false;
        }
        if transpose_a || transpose_b {
            return false;
        }
        if dims.batch != 1 {
            return false;
        }
        dims.m == 1
    }

    #[inline]
    fn can_use_q8_nt(&self, dims: &MatmulDims, transpose_a: bool) -> bool {
        if transpose_a {
            return false;
        }
        if dims.batch != 1 {
            return false;
        }
        dims.m > 0 && dims.m <= MATMUL_Q8_NT_MAX_ROWS
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        &mut self,
        a: &Tensor<T>,
        b_any: &TensorType<T>,
        transpose_a: bool,
        transpose_b: bool,
        bias: Option<&Tensor<T>>,
        alpha_beta: Option<MatmulAlphaBeta<'_, T>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<Tensor<T>, MetalError> {
        if bias.is_some() && alpha_beta.is_some() {
            todo!("bias + alpha/beta epilogues are not implemented yet");
        }

        let mut cache = cache;

        match b_any {
            TensorType::Dense(bd) => self.matmul_dense(a, bd, transpose_a, transpose_b, bias, alpha_beta, cache.as_deref_mut()),
            TensorType::DenseCanonical(bc) => {
                self.matmul_dense_canonical(a, bc, transpose_a, transpose_b, bias, alpha_beta, cache.as_deref_mut())
            }
            TensorType::Quant(qrhs) => self.matmul_quant(a, qrhs, transpose_a, transpose_b, bias, alpha_beta, cache.as_deref_mut()),
        }
    }

    fn matmul_dense_canonical(
        &mut self,
        a: &Tensor<T>,
        b: &CanonicalF16Tensor<T>,
        transpose_a: bool,
        transpose_b: bool,
        bias: Option<&Tensor<T>>,
        alpha_beta: Option<MatmulAlphaBeta<'_, T>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<Tensor<T>, MetalError> {
        let mut cache = cache;
        if T::DTYPE != crate::tensor::Dtype::F16 {
            return Err(MetalError::UnsupportedDtype {
                operation: "dense canonical matmul",
                dtype: T::DTYPE,
            });
        }
        if transpose_b {
            return Err(MetalError::OperationNotSupported(
                "canonical matmul does not support transpose_b".into(),
            ));
        }

        let rhs = TensorType::DenseCanonical(b);
        let dims = self.compute_matmul_dims(a, &rhs, transpose_a, transpose_b)?;
        if dims.m == 1 {
            if !self.can_use_gemv_canonical(&dims, transpose_a, transpose_b) {
                emit_matmul_backend_selected(
                    "dense_canonical",
                    Some(dims),
                    transpose_a,
                    transpose_b,
                    "Unsupported",
                    "canonical_requires_gemv_m1",
                );
                return Err(MetalError::OperationNotSupported(
                    "canonical matmul only supports GEMV m=1 without transposes".into(),
                ));
            }

            emit_matmul_backend_selected("dense_canonical", Some(dims), transpose_a, transpose_b, "Gemv", "m1");
            return self.launch_gemv(a, rhs, transpose_b, bias, alpha_beta, cache.as_deref_mut());
        }

        if transpose_a || transpose_b || dims.batch != 1 {
            emit_matmul_backend_selected(
                "dense_canonical",
                Some(dims),
                transpose_a,
                transpose_b,
                "Unsupported",
                "canonical_requires_no_transpose_batch1",
            );
            return Err(MetalError::OperationNotSupported(
                "canonical GEMM requires batch=1 and no transposes".into(),
            ));
        }

        if alpha_beta.is_some() {
            emit_matmul_backend_selected(
                "dense_canonical",
                Some(dims),
                transpose_a,
                transpose_b,
                "Unsupported",
                "canonical_gemm_no_alpha_beta",
            );
            return Err(MetalError::OperationNotSupported(
                "canonical GEMM does not support alpha/beta epilogue yet".into(),
            ));
        }

        if self.should_use_f16_canonical_rows16(&dims, transpose_a, transpose_b) {
            emit_matmul_backend_selected(
                "dense_canonical",
                Some(dims),
                transpose_a,
                transpose_b,
                "F16CanonicalRows16",
                "heuristic",
            );
            return self.call::<MatmulF16CanonicalRows16Op>((a, b, bias), cache.as_deref_mut());
        }

        emit_matmul_backend_selected("dense_canonical", Some(dims), transpose_a, transpose_b, "F16Canonical", "heuristic");
        self.call::<MatmulF16CanonicalOp>((a, b, bias), cache.as_deref_mut())
    }

    pub fn qkv(&mut self, x_flat: &Tensor<T>, weights: QkvWeights<'_, T>) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>), MetalError> {
        self.with_gpu_scope("attn_qkv_proj".to_string(), |ctx| match weights {
            QkvWeights::Dense {
                fused_weight,
                fused_bias,
                d_model,
                kv_dim,
            } => ctx.qkv_dense(x_flat, fused_weight, fused_bias, d_model, kv_dim),
            QkvWeights::Quantized {
                wq,
                wk,
                wv,
                q_bias,
                k_bias,
                v_bias,
            } => ctx.qkv_quant(x_flat, wq, wk, wv, q_bias, k_bias, v_bias),
            QkvWeights::DenseCanonical {
                wq,
                wk,
                wv,
                q_bias,
                k_bias,
                v_bias,
            } => ctx.qkv_dense_canonical(x_flat, wq, wk, wv, q_bias, k_bias, v_bias),
        })
    }

    fn matmul_dense(
        &mut self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        transpose_a: bool,
        transpose_b: bool,
        bias: Option<&Tensor<T>>,
        alpha_beta: Option<MatmulAlphaBeta<'_, T>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<Tensor<T>, MetalError> {
        let mut cache = cache;
        let rhs = TensorType::Dense(b);
        let dims_result = self.compute_matmul_dims(a, &rhs, transpose_a, transpose_b);
        let dims_ok = match &dims_result {
            Ok(d) => Some(*d),
            Err(_) => None,
        };
        // Enable GEMV even if bias/alpha_beta present (our kernels support them)
        let can_gemv = dims_ok
            .as_ref()
            .map(|dims| self.can_use_gemv(dims, transpose_a, transpose_b))
            .unwrap_or(false);

        match self.forced_matmul_backend {
            InternalMatMulBackendOverride::Force(MatMulBackend::Mlx) => {
                emit_matmul_backend_selected("dense", dims_ok, transpose_a, transpose_b, "Mlx", "forced_backend");
                return self.matmul_dense_mlx(a, b, transpose_a, transpose_b, bias, alpha_beta, cache.as_deref_mut());
            }
            InternalMatMulBackendOverride::Force(MatMulBackend::Gemv) => {
                if can_gemv {
                    emit_matmul_backend_selected("dense", dims_ok, transpose_a, transpose_b, "Gemv", "forced_backend");
                    return self.launch_gemv(a, TensorType::Dense(b), transpose_b, bias, alpha_beta, cache.as_deref_mut());
                }
                emit_matmul_backend_selected("dense", dims_ok, transpose_a, transpose_b, "Mlx", "forced_backend_gemv_unsupported");
                return self.matmul_dense_mlx(a, b, transpose_a, transpose_b, bias, alpha_beta, cache.as_deref_mut());
            }
            InternalMatMulBackendOverride::Force(MatMulBackend::Mps) => {
                if can_gemv {
                    emit_matmul_backend_selected("dense", dims_ok, transpose_a, transpose_b, "Gemv", "forced_backend");
                    return self.launch_gemv(a, TensorType::Dense(b), transpose_b, bias, alpha_beta, cache.as_deref_mut());
                }
                emit_matmul_backend_selected("dense", dims_ok, transpose_a, transpose_b, "Mps", "forced_backend");
                return self.matmul_dense_mps(a, b, transpose_a, transpose_b, bias, alpha_beta, cache.as_deref_mut());
            }
            InternalMatMulBackendOverride::Default | InternalMatMulBackendOverride::Auto => {}
        }

        if can_gemv {
            emit_matmul_backend_selected("dense", dims_ok, transpose_a, transpose_b, "Gemv", "heuristic_m1");
            return self.launch_gemv(a, TensorType::Dense(b), transpose_b, bias, alpha_beta, cache.as_deref_mut());
        }

        let dims = match dims_result {
            Ok(d) => d,
            Err(_) => {
                emit_matmul_backend_selected("dense", dims_ok, transpose_a, transpose_b, "Mlx", "dims_inference_failed");
                return self.matmul_dense_mlx(a, b, transpose_a, transpose_b, bias, alpha_beta, cache.as_deref_mut());
            }
        };

        let has_strided_batch = {
            let mut tensors: Vec<&Tensor<T>> = vec![a, b];
            if let Some(ep) = alpha_beta {
                tensors.push(ep.output);
            }
            self.has_strided_mps_batch(&tensors)
        };

        let use_mlx = if alpha_beta.is_some() {
            self.should_use_mlx_dense(&dims, has_strided_batch)
        } else if bias.is_some() {
            self.should_use_mlx_bias(&dims)
        } else {
            self.should_use_mlx_dense(&dims, has_strided_batch)
        };

        if use_mlx {
            emit_matmul_backend_selected(
                "dense",
                Some(dims),
                transpose_a,
                transpose_b,
                "Mlx",
                if bias.is_some() {
                    "heuristic_bias"
                } else if alpha_beta.is_some() {
                    "heuristic_alpha_beta"
                } else {
                    "heuristic"
                },
            );
            self.matmul_dense_mlx(a, b, transpose_a, transpose_b, bias, alpha_beta, cache.as_deref_mut())
        } else {
            emit_matmul_backend_selected(
                "dense",
                Some(dims),
                transpose_a,
                transpose_b,
                "Mps",
                if bias.is_some() {
                    "heuristic_bias"
                } else if alpha_beta.is_some() {
                    "heuristic_alpha_beta"
                } else {
                    "heuristic"
                },
            );
            self.matmul_dense_mps(a, b, transpose_a, transpose_b, bias, alpha_beta, cache)
        }
    }

    fn matmul_dense_mlx(
        &mut self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        transpose_a: bool,
        transpose_b: bool,
        bias: Option<&Tensor<T>>,
        alpha_beta: Option<MatmulAlphaBeta<'_, T>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<Tensor<T>, MetalError> {
        let _ = cache; // MLX currently doesn't use cache
        let (existing_out, alpha, beta) = match alpha_beta {
            Some(ep) => (Some(ep.output), ep.alpha, ep.beta),
            None => (None, 1.0, 0.0),
        };
        self.call::<MatMulMlxOp>(
            (a, TensorType::Dense(b), bias, existing_out, transpose_a, transpose_b, alpha, beta),
            None,
        )
    }

    fn matmul_dense_mps(
        &mut self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        transpose_a: bool,
        transpose_b: bool,
        bias: Option<&Tensor<T>>,
        alpha_beta: Option<MatmulAlphaBeta<'_, T>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<Tensor<T>, MetalError> {
        let mut cache = cache;
        if let Some(ep) = alpha_beta {
            return self.call::<MatMulMpsAlphaBetaOp>((a, b, ep.output, transpose_a, transpose_b, ep.alpha, ep.beta), cache.as_deref_mut());
        }

        let mut out = self.call::<MatMulMpsOp>((a, b, transpose_a, transpose_b), cache.as_deref_mut())?;
        if let Some(bias_tensor) = bias {
            // Bias add does not benefit from external cache; use internal path to avoid extra CB path
            out = self.call::<BroadcastElemwiseAddInplaceOp>((out, bias_tensor.clone()), None)?;
        }
        Ok(out)
    }

    fn launch_gemv(
        &mut self,
        a: &Tensor<T>,
        rhs: TensorType<'_, T>,
        transpose_right: bool,
        bias: Option<&Tensor<T>>,
        alpha_beta: Option<MatmulAlphaBeta<'_, T>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<Tensor<T>, MetalError> {
        let mut cache = cache;
        if let Some(ep) = alpha_beta {
            self.call::<MatmulGemvAddmmOp>(
                (a, rhs, bias, Some(ep.output), transpose_right, ep.alpha, ep.beta),
                cache.as_deref_mut(),
            )
        } else {
            self.call::<MatmulGemvOp>((a, rhs, transpose_right, bias), cache.as_deref_mut())
        }
    }

    fn matmul_quant(
        &mut self,
        a: &Tensor<T>,
        qrhs: &QuantizedTensor<'_>,
        transpose_a: bool,
        transpose_b: bool,
        bias: Option<&Tensor<T>>,
        alpha_beta: Option<MatmulAlphaBeta<'_, T>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<Tensor<T>, MetalError> {
        let mut cache = cache;
        if alpha_beta.is_some() {
            todo!("quantized matmul with alpha/beta is not implemented");
        }

        if T::DTYPE != crate::tensor::Dtype::F16 {
            return Err(MetalError::UnsupportedDtype {
                operation: "quant matmul",
                dtype: T::DTYPE,
            });
        }

        match qrhs {
            QuantizedTensor::Q8_0(q8) => {
                let rhs = TensorType::Quant(QuantizedTensor::Q8_0(q8));
                let dims = self.compute_matmul_dims(a, &rhs, transpose_a, transpose_b)?;
                if transpose_b {
                    if transpose_a {
                        return Err(MetalError::OperationNotSupported(
                            "quant matmul: transpose_a+transpose_b unsupported".into(),
                        ));
                    }

                    if dims.batch == 1 && dims.m == 1 {
                        emit_matmul_backend_selected("q8_0", Some(dims), transpose_a, transpose_b, "Gemv", "m1");
                        return self.call::<MatmulGemvOp>(
                            (a, TensorType::Quant(QuantizedTensor::Q8_0(q8)), transpose_b, bias),
                            cache.as_deref_mut(),
                        );
                    }
                    if self.should_use_q8_canonical_rows16(&dims, transpose_a) {
                        emit_matmul_backend_selected("q8_0", Some(dims), transpose_a, transpose_b, "Q8CanonicalRows16", "heuristic");
                        return self.call::<MatmulQ8CanonicalRows16Op>((a, q8, bias), cache.as_deref_mut());
                    }
                    if self.should_use_q8_canonical(&dims, transpose_a) {
                        emit_matmul_backend_selected("q8_0", Some(dims), transpose_a, transpose_b, "Q8Canonical", "heuristic");
                        return self.call::<MatmulQ8CanonicalOp>((a, q8, bias), cache.as_deref_mut());
                    }
                    let smallm_use_gemv = std::env::var("METALLIC_Q8_SMALLM_USE_GEMV").ok().map(|s| s != "0").unwrap_or(true);
                    let smallm_fallback_max_n = Self::smallm_fallback_max_n();
                    if smallm_use_gemv && dims.batch == 1 && (2..=4).contains(&dims.m) && !transpose_a && dims.n <= smallm_fallback_max_n {
                        emit_matmul_backend_selected("q8_0", Some(dims), transpose_a, transpose_b, "GemvSmallM", "heuristic");
                        return self.call::<MatmulGemvSmallMOp>((a, q8, bias), cache.as_deref_mut());
                    }
                    if self.can_use_q8_nt(&dims, transpose_a) {
                        emit_matmul_backend_selected("q8_0", Some(dims), transpose_a, transpose_b, "Q8Nt", "heuristic");
                        return self.call::<MatmulQ8NtOp>((a, q8, bias), cache.as_deref_mut());
                    }
                    emit_matmul_backend_selected("q8_0", Some(dims), transpose_a, transpose_b, "Unsupported", "no_supported_kernel");
                    return Err(MetalError::OperationNotSupported(
                        "quant matmul with transpose_b=true is not supported for this shape (try keeping weights in [K,N] and calling with transpose_b=false, or reduce m<=4 so MatmulQ8Nt can be used)".into(),
                    ));
                }

                if dims.batch == 1 && dims.m == 1 && !transpose_a {
                    if q8_should_use_mlx_for_m1(&dims, transpose_a, transpose_b) {
                        emit_matmul_backend_selected("q8_0", Some(dims), transpose_a, transpose_b, "Mlx", "heuristic_m1_large_n");
                        return self.call::<MatMulMlxOp>(
                            (
                                a,
                                TensorType::Quant(QuantizedTensor::Q8_0(q8)),
                                bias,
                                None,
                                transpose_a,
                                false,
                                1.0,
                                0.0,
                            ),
                            None,
                        );
                    }
                    emit_matmul_backend_selected("q8_0", Some(dims), transpose_a, transpose_b, "Gemv", "m1");
                    return self.call::<MatmulGemvOp>(
                        (a, TensorType::Quant(QuantizedTensor::Q8_0(q8)), transpose_b, bias),
                        cache.as_deref_mut(),
                    );
                }
                if dims.batch == 1 && (2..=4).contains(&dims.m) && !transpose_a {
                    if self.should_use_q8_canonical_rows16(&dims, transpose_a) {
                        emit_matmul_backend_selected("q8_0", Some(dims), transpose_a, transpose_b, "Q8CanonicalRows16", "heuristic");
                        return self.call::<MatmulQ8CanonicalRows16Op>((a, q8, bias), cache.as_deref_mut());
                    }
                    emit_matmul_backend_selected("q8_0", Some(dims), transpose_a, transpose_b, "Q8Canonical", "heuristic_smallm");
                    return self.call::<MatmulQ8CanonicalOp>((a, q8, bias), cache.as_deref_mut());
                }
                if self.should_use_q8_canonical(&dims, transpose_a) {
                    emit_matmul_backend_selected("q8_0", Some(dims), transpose_a, transpose_b, "Q8Canonical", "heuristic");
                    return self.call::<MatmulQ8CanonicalOp>((a, q8, bias), cache.as_deref_mut());
                }
                emit_matmul_backend_selected("q8_0", Some(dims), transpose_a, transpose_b, "Mlx", "fallback");
                self.call::<MatMulMlxOp>(
                    (
                        a,
                        TensorType::Quant(QuantizedTensor::Q8_0(q8)),
                        bias,
                        None,
                        transpose_a,
                        false,
                        1.0,
                        0.0,
                    ),
                    None,
                )
            }
        }
    }

    fn qkv_dense(
        &mut self,
        x_flat: &Tensor<T>,
        fused_weight: &Tensor<T>,
        fused_bias: &Tensor<T>,
        d_model: usize,
        kv_dim: usize,
    ) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>), MetalError> {
        let x_dims = x_flat.dims();
        if x_dims.len() != 2 {
            return Err(MetalError::InvalidShape(format!(
                "qkv expects a 2D input [m, d_model], got {:?}",
                x_dims
            )));
        }

        let in_features = x_dims[1];
        if in_features != d_model {
            return Err(MetalError::InvalidShape(format!(
                "Input feature size {} does not match d_model {}",
                in_features, d_model
            )));
        }

        let weight_dims = fused_weight.dims();
        if weight_dims.len() != 2 {
            return Err(MetalError::InvalidShape(format!(
                "Fused weight must be 2D [d_model, qkv], got {:?}",
                weight_dims
            )));
        }

        let expected_total = d_model + 2 * kv_dim;
        if weight_dims[0] != d_model || weight_dims[1] != expected_total {
            return Err(MetalError::InvalidShape(format!(
                "Fused weight dims {:?} incompatible with d_model {} and kv_dim {}",
                weight_dims, d_model, kv_dim
            )));
        }

        if fused_bias.dims() != [expected_total] {
            return Err(MetalError::InvalidShape(format!(
                "Fused bias dims {:?} incompatible with expected total {}",
                fused_bias.dims(),
                expected_total
            )));
        }

        let linear = self.matmul(x_flat, &TensorType::Dense(fused_weight), false, false, Some(fused_bias), None, None)?;

        let q_range_end = d_model;
        let k_range_end = d_model + kv_dim;
        let v_range_end = expected_total;

        let q_out = linear.slice_last_dim(0..q_range_end)?;
        let k_out = linear.slice_last_dim(d_model..k_range_end)?;
        let v_out = linear.slice_last_dim(k_range_end..v_range_end)?;

        Ok((q_out, k_out, v_out))
    }

    fn qkv_quant(
        &mut self,
        x_flat: &Tensor<T>,
        wq_q8: &crate::tensor::QuantizedQ8_0Tensor,
        wk_q8: &crate::tensor::QuantizedQ8_0Tensor,
        wv_q8: &crate::tensor::QuantizedQ8_0Tensor,
        q_bias: &Tensor<T>,
        k_bias: &Tensor<T>,
        v_bias: &Tensor<T>,
    ) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>), MetalError> {
        if T::DTYPE != crate::tensor::Dtype::F16 {
            return Err(MetalError::UnsupportedDtype {
                operation: "qkv/quant",
                dtype: T::DTYPE,
            });
        }

        let x_dims = x_flat.dims();
        if x_dims.len() != 2 || x_dims[0] != 1 {
            return Err(MetalError::InvalidShape(format!(
                "qkv(quant) expects x_flat=[1,K], got {:?}",
                x_dims
            )));
        }
        let k = x_dims[1];
        let wq = &wq_q8.logical_dims;
        let wk = &wk_q8.logical_dims;
        let wv = &wv_q8.logical_dims;
        if wq.get(0) != Some(&k) || wk.get(0) != Some(&k) || wv.get(0) != Some(&k) {
            return Err(MetalError::InvalidShape("QKV K mismatch for quant qkv".into()));
        }
        let nq = *wq.get(1).ok_or(MetalError::InvalidShape("wq dims".into()))?;
        let nk = *wk.get(1).ok_or(MetalError::InvalidShape("wk dims".into()))?;
        let nv = *wv.get(1).ok_or(MetalError::InvalidShape("wv dims".into()))?;

        let y = self.call::<MatmulGemvQkvFusedOp>(
            (
                x_flat,
                (
                    &QuantizedTensor::Q8_0(wq_q8),
                    &QuantizedTensor::Q8_0(wk_q8),
                    &QuantizedTensor::Q8_0(wv_q8),
                ),
                (Some(q_bias), Some(k_bias), Some(v_bias)),
            ),
            None,
        )?;

        let elem = T::DTYPE.size_bytes();
        let q_out = y.build_view(vec![1, nq], vec![nq, 1], y.offset);
        let k_out = y.build_view(vec![1, nk], vec![nk, 1], y.offset + nq * elem);
        let v_out = y.build_view(vec![1, nv], vec![nv, 1], y.offset + (nq + nk) * elem);

        Ok((q_out, k_out, v_out))
    }

    fn qkv_dense_canonical(
        &mut self,
        x_flat: &Tensor<T>,
        wq: &CanonicalF16Tensor<T>,
        wk: &CanonicalF16Tensor<T>,
        wv: &CanonicalF16Tensor<T>,
        q_bias: &Tensor<T>,
        k_bias: &Tensor<T>,
        v_bias: &Tensor<T>,
    ) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>), MetalError> {
        if T::DTYPE != crate::tensor::Dtype::F16 {
            return Err(MetalError::UnsupportedDtype {
                operation: "qkv/dense_canonical",
                dtype: T::DTYPE,
            });
        }

        // Ensure we are in a supported shape/config for canonical fused kernel
        // e.g. batch=1, m=1. For now, canonical kernels are strict about this.
        let x_dims = x_flat.dims();
        if x_dims.len() != 2 || x_dims[0] != 1 {
            // Fallback? Or generic Matmul?
            // println!("Fallback: qkv_dense_canonical called with x_dims={:?}", x_dims);
        } else {
            // println!("HIT: qkv_dense_canonical");
        }

        let (yq, yk, yv) =
            self.call_custom::<MatmulF16CanonicalQkvFusedOp>((x_flat, (wq, wk, wv), (Some(q_bias), Some(k_bias), Some(v_bias))), None)?;

        Ok((yq, yk, yv))
    }

    pub fn swiglu(
        &mut self,
        x: &Tensor<T>,
        gate: &crate::tensor::TensorType<T>,
        up: &crate::tensor::TensorType<T>,
        gate_bias: Option<&Tensor<T>>,
        bias_down: Option<&Tensor<T>>,
    ) -> Result<Tensor<T>, MetalError> {
        match (gate, up) {
            (crate::tensor::TensorType::DenseCanonical(wg), crate::tensor::TensorType::DenseCanonical(wu)) => {
                self.call::<MatmulF16CanonicalSwiGluOp>((x, (wg, wu), (gate_bias, bias_down)), None)
            }
            _ => Err(MetalError::InvalidOperation(
                "SwiGLU only supported for DenseCanonical F16 currently".into(),
            )),
        }
    }
}
