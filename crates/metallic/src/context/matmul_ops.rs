use super::{main::Context, utils::MatMulBackendOverride as InternalMatMulBackendOverride};
use crate::{
    MetalError, Tensor, caching::ResourceCache, kernels::{
        elemwise_add::BroadcastElemwiseAddInplaceOp, matmul_gemv::{MatmulGemvAddmmOp, MatmulGemvOp}, matmul_gemv_qkv_fused::MatmulGemvQkvFusedOp, matmul_mlx::MatMulMlxOp, matmul_mps::{MatMulBackend, MatMulMpsAlphaBetaOp, MatMulMpsOp}, matmul_q8_canonical::{MatmulQ8CanonicalOp, MatmulQ8CanonicalRows16Op}, matmul_q8_nt::{MATMUL_Q8_NT_MAX_ROWS, MatmulQ8NtOp}
    }, tensor::{QuantizedTensor, TensorElement, TensorType}
};

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
            TensorType::Quant(qrhs) => self.matmul_quant(a, qrhs, transpose_a, transpose_b, bias, alpha_beta, cache.as_deref_mut()),
        }
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
        // Match previous behavior: only consider GEMV for pure matmul (no bias, no alpha/beta)
        let can_gemv = alpha_beta.is_none()
            && bias.is_none()
            && dims_ok
                .as_ref()
                .map(|dims| self.can_use_gemv(dims, transpose_a, transpose_b))
                .unwrap_or(false);

        match self.forced_matmul_backend {
            InternalMatMulBackendOverride::Force(MatMulBackend::Mlx) => {
                return self.matmul_dense_mlx(a, b, transpose_a, transpose_b, bias, alpha_beta, cache.as_deref_mut());
            }
            InternalMatMulBackendOverride::Force(MatMulBackend::Gemv) => {
                if can_gemv {
                    return self.launch_gemv(a, TensorType::Dense(b), bias, alpha_beta, cache.as_deref_mut());
                }
                return self.matmul_dense_mlx(a, b, transpose_a, transpose_b, bias, alpha_beta, cache.as_deref_mut());
            }
            InternalMatMulBackendOverride::Force(MatMulBackend::Mps) => {
                if can_gemv {
                    return self.launch_gemv(a, TensorType::Dense(b), bias, alpha_beta, cache.as_deref_mut());
                }
                return self.matmul_dense_mps(a, b, transpose_a, transpose_b, bias, alpha_beta, cache.as_deref_mut());
            }
            InternalMatMulBackendOverride::Default | InternalMatMulBackendOverride::Auto => {}
        }

        if can_gemv {
            return self.launch_gemv(a, TensorType::Dense(b), bias, alpha_beta, cache.as_deref_mut());
        }

        let dims = match dims_result {
            Ok(d) => d,
            Err(_) => return self.matmul_dense_mlx(a, b, transpose_a, transpose_b, bias, alpha_beta, cache.as_deref_mut()),
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
            self.matmul_dense_mlx(a, b, transpose_a, transpose_b, bias, alpha_beta, cache.as_deref_mut())
        } else {
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
        bias: Option<&Tensor<T>>,
        alpha_beta: Option<MatmulAlphaBeta<'_, T>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<Tensor<T>, MetalError> {
        let mut cache = cache;
        if let Some(ep) = alpha_beta {
            self.call::<MatmulGemvAddmmOp>((a, rhs, bias, Some(ep.output), ep.alpha, ep.beta), cache.as_deref_mut())
        } else {
            self.call::<MatmulGemvOp>((a, rhs, bias), cache.as_deref_mut())
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

                    if dims.batch == 1 && dims.m == 1 && dims.k <= 1024 && dims.n >= 2048 {
                        return self.call::<MatMulMlxOp>(
                            (a, TensorType::Quant(QuantizedTensor::Q8_0(q8)), bias, None, false, false, 1.0, 0.0),
                            cache.as_deref_mut(),
                        );
                    }

                    if dims.batch == 1 && dims.m == 1 {
                        return self.call::<MatmulGemvOp>((a, TensorType::Quant(QuantizedTensor::Q8_0(q8)), bias), cache.as_deref_mut());
                    }
                    if self.should_use_q8_canonical_rows16(&dims, transpose_a) {
                        return self.call::<MatmulQ8CanonicalRows16Op>((a, q8, bias), cache.as_deref_mut());
                    }
                    if self.should_use_q8_canonical(&dims, transpose_a) {
                        return self.call::<MatmulQ8CanonicalOp>((a, q8, bias), cache.as_deref_mut());
                    }
                    let smallm_use_gemv = std::env::var("METALLIC_Q8_SMALLM_USE_GEMV").ok().map(|s| s != "0").unwrap_or(true);
                    let smallm_fallback_max_n = Self::smallm_fallback_max_n();
                    if smallm_use_gemv && dims.batch == 1 && (2..=4).contains(&dims.m) && !transpose_a && dims.n <= smallm_fallback_max_n {
                        return self.call::<crate::kernels::matmul_gemv_smallm::MatmulGemvSmallMOp>((a, q8, bias), cache.as_deref_mut());
                    }
                    if self.can_use_q8_nt(&dims, transpose_a) {
                        return self.call::<MatmulQ8NtOp>((a, q8, bias), cache.as_deref_mut());
                    }
                    return self.call::<MatMulMlxOp>(
                        (
                            a,
                            TensorType::Quant(QuantizedTensor::Q8_0(q8)),
                            bias,
                            None,
                            transpose_a,
                            transpose_b,
                            1.0,
                            0.0,
                        ),
                        None,
                    );
                }

                if dims.batch == 1 && dims.m == 1 && !transpose_a {
                    return self.call::<MatmulGemvOp>((a, TensorType::Quant(QuantizedTensor::Q8_0(q8)), bias), cache.as_deref_mut());
                }
                if dims.batch == 1 && (2..=4).contains(&dims.m) && !transpose_a {
                    if self.should_use_q8_canonical_rows16(&dims, transpose_a) {
                        return self.call::<MatmulQ8CanonicalRows16Op>((a, q8, bias), cache.as_deref_mut());
                    }
                    return self.call::<MatmulQ8CanonicalOp>((a, q8, bias), cache.as_deref_mut());
                }
                if self.should_use_q8_canonical(&dims, transpose_a) {
                    return self.call::<MatmulQ8CanonicalOp>((a, q8, bias), cache.as_deref_mut());
                }
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
}
