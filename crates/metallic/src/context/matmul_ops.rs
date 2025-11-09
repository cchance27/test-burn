use super::{main::Context, utils::MatMulBackendOverride as InternalMatMulBackendOverride};
use crate::{
    MetalError, Tensor, caching::ResourceCache, kernels::{
        elemwise_add::BroadcastElemwiseAddInplaceOp, matmul_gemv::MatmulGemvOp, matmul_gemv_qkv_fused::MatmulGemvQkvFusedOp, matmul_mlx::MatMulMlxOp, matmul_mps::{MatMulBackend, MatMulMpsAlphaBetaOp, MatMulMpsOp}, matmul_q8_canonical::{MatmulQ8CanonicalOp, MatmulQ8CanonicalRows16Op}, matmul_q8_nt::{MATMUL_Q8_NT_MAX_ROWS, MatmulQ8NtOp}
    }, tensor::{QuantizedTensor, TensorElement, TensorType}
};

#[derive(Debug, Clone, Copy)]
pub struct MatmulDims {
    pub batch: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl<T: TensorElement> Context<T> {
    fn smallm_fallback_max_n() -> usize {
        std::env::var("METALLIC_Q8_SMALLM_FALLBACK_MAX_N")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4096)
    }

    fn canonical_force_enabled() -> bool {
        std::env::var("METALLIC_Q8_CANONICAL_N").ok().map(|s| s != "0").unwrap_or(false)
    }

    fn canonical_min_n_threshold() -> usize {
        std::env::var("METALLIC_Q8_CANONICAL_MIN_N")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4096)
    }

    fn canonical_max_m() -> usize {
        std::env::var("METALLIC_Q8_CANONICAL_MAX_M")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(16)
    }

    // Reserved for future tunables.

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
        if let Ok(s) = std::env::var("METALLIC_Q8_CANONICAL_R16_MAX_N") {
            if let Ok(val) = s.parse::<usize>() {
                return val;
            }
        }
        match m {
            2 | 3 | 4 => 1536,
            _ => 0,
        }
    }

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

    fn can_use_q8_nt(&self, dims: &MatmulDims, transpose_a: bool) -> bool {
        if transpose_a {
            return false;
        }
        if dims.batch != 1 {
            return false;
        }
        dims.m > 0 && dims.m <= MATMUL_Q8_NT_MAX_ROWS
    }

    fn call_mps_matmul_op(
        &mut self,
        args: (&Tensor<T>, &Tensor<T>, bool, bool),
        cache: Option<&mut ResourceCache>,
    ) -> Result<Tensor<T>, MetalError> {
        if let Some(cache) = cache {
            self.call_with_cache::<MatMulMpsOp>((args.0, args.1, args.2, args.3), cache)
        } else {
            self.call::<MatMulMpsOp>((args.0, args.1, args.2, args.3))
        }
    }

    fn call_mps_alpha_beta_op(
        &mut self,
        args: (&Tensor<T>, &Tensor<T>, &Tensor<T>, bool, bool, f32, f32),
        cache: Option<&mut ResourceCache>,
    ) -> Result<Tensor<T>, MetalError> {
        if let Some(cache) = cache {
            self.call_with_cache::<MatMulMpsAlphaBetaOp>(args, cache)
        } else {
            self.call::<MatMulMpsAlphaBetaOp>(args)
        }
    }

    // Quant-aware variant: accepts dense or quantized RHS. For now, only GEMV on quant is supported via proxy op.
    #[inline]
    pub fn matmul(
        &mut self,
        a: &Tensor<T>,
        b_any: &TensorType<T>,
        transpose_a: bool,
        transpose_b: bool,
        cache: Option<&mut ResourceCache>,
    ) -> Result<Tensor<T>, MetalError> {
        // TODO: Add cache support since we dropped the legacy matmul_with_cache route
        match b_any {
            TensorType::Dense(bd) => match self.forced_matmul_backend {
                InternalMatMulBackendOverride::Force(backend) => match backend {
                    MatMulBackend::Mlx => {
                        self.call::<MatMulMlxOp>((a, TensorType::Dense(bd), None, None, transpose_a, transpose_b, 1.0, 0.0))
                    }
                    MatMulBackend::Mps => {
                        let dims_result = self.compute_matmul_dims(a, b_any, transpose_a, transpose_b);
                        match dims_result {
                            Ok(dimensions) => {
                                if self.can_use_gemv(&dimensions, transpose_a, transpose_b) {
                                    self.call::<MatmulGemvOp>((a, TensorType::Dense(bd), None))
                                } else {
                                    self.call_mps_matmul_op((a, bd, transpose_a, transpose_b), cache)
                                }
                            }
                            Err(_) => self.call::<MatMulMlxOp>((a, TensorType::Dense(bd), None, None, transpose_a, transpose_b, 1.0, 0.0)),
                        }
                    }
                    MatMulBackend::Gemv => {
                        let dims_result = self.compute_matmul_dims(a, b_any, transpose_a, transpose_b);
                        match dims_result {
                            Ok(dimensions) => {
                                if self.can_use_gemv(&dimensions, transpose_a, transpose_b) {
                                    self.call::<MatmulGemvOp>((a, TensorType::Dense(bd), None))
                                } else {
                                    self.call::<MatMulMlxOp>((a, TensorType::Dense(bd), None, None, transpose_a, transpose_b, 1.0, 0.0))
                                }
                            }
                            Err(_) => self.call::<MatMulMlxOp>((a, TensorType::Dense(bd), None, None, transpose_a, transpose_b, 1.0, 0.0)),
                        }
                    }
                },
                InternalMatMulBackendOverride::Default | InternalMatMulBackendOverride::Auto => {
                    let dims_result = self.compute_matmul_dims(a, b_any, transpose_a, transpose_b);

                    match dims_result {
                        Ok(dimensions) => {
                            if self.can_use_gemv(&dimensions, transpose_a, transpose_b) {
                                return self.call::<MatmulGemvOp>((a, TensorType::Dense(bd), None));
                            }

                            let has_strided_batch = self.has_strided_mps_batch(&[a, bd]);
                            let use_mlx = self.should_use_mlx_dense(&dimensions, has_strided_batch);

                            if use_mlx {
                                self.call::<MatMulMlxOp>((a, TensorType::Dense(bd), None, None, transpose_a, transpose_b, 1.0, 0.0))
                            } else {
                                self.call_mps_matmul_op((a, bd, transpose_a, transpose_b), cache)
                            }
                        }
                        Err(_) => self.call::<MatMulMlxOp>((a, TensorType::Dense(bd), None, None, transpose_a, transpose_b, 1.0, 0.0)),
                    }
                }
            },
            TensorType::Quant(qrhs) => {
                let dims = self.compute_matmul_dims(a, b_any, transpose_a, transpose_b)?;
                if T::DTYPE != crate::tensor::Dtype::F16 {
                    return Err(MetalError::UnsupportedDtype {
                        operation: "quant matmul",
                        dtype: T::DTYPE,
                    });
                }
                if transpose_b {
                    if transpose_a {
                        return Err(MetalError::OperationNotSupported(
                            "quant matmul: transpose_a+transpose_b unsupported".into(),
                        ));
                    }
                    // Decode optimization: for tall-N cases where K is small, MLX Q8 (tB=false) is much faster than GEMV.
                    // Use MLX quant path by virtually flipping tB to false when K is small and N is large.
                    if dims.batch == 1 && dims.m == 1 && dims.k <= 1024 && dims.n >= 2048 {
                        return match qrhs {
                            QuantizedTensor::Q8_0(q8) => self.call::<crate::kernels::matmul_mlx::MatMulMlxOp>((
                                a,
                                TensorType::Quant(QuantizedTensor::Q8_0(q8)),
                                None,
                                None,
                                false, // treat RHS as not transposed for MLX quant loader
                                false,
                                1.0,
                                0.0,
                            )),
                        };
                    }
                    // For decode (m=1), GEMV consistently outperforms NT in benchmarks.
                    if dims.batch == 1 && dims.m == 1 {
                        return match qrhs {
                            QuantizedTensor::Q8_0(q8) => {
                                self.call::<MatmulGemvOp>((a, TensorType::Quant(QuantizedTensor::Q8_0(q8)), None))
                            }
                        };
                    }
                    if self.should_use_q8_canonical_rows16(&dims, transpose_a) {
                        return match qrhs {
                            QuantizedTensor::Q8_0(q8) => self.call::<MatmulQ8CanonicalRows16Op>((a, q8, None)),
                        };
                    }
                    if self.should_use_q8_canonical(&dims, transpose_a) {
                        return match qrhs {
                            QuantizedTensor::Q8_0(q8) => self.call::<MatmulQ8CanonicalOp>((a, q8, None)),
                        };
                    }
                    // For small m (2..=4), prefer the small-m kernel where allowed.
                    let smallm_use_gemv = std::env::var("METALLIC_Q8_SMALLM_USE_GEMV").ok().map(|s| s != "0").unwrap_or(true);
                    let smallm_fallback_max_n = Self::smallm_fallback_max_n();
                    if smallm_use_gemv && dims.batch == 1 && (2..=4).contains(&dims.m) && !transpose_a && dims.n <= smallm_fallback_max_n {
                        return match qrhs {
                            QuantizedTensor::Q8_0(q8) => self.call::<crate::kernels::matmul_gemv_smallm::MatmulGemvSmallMOp>((a, q8, None)),
                        };
                    }
                    if self.can_use_q8_nt(&dims, transpose_a) {
                        return match qrhs {
                            QuantizedTensor::Q8_0(q8) => self.call::<MatmulQ8NtOp>((a, q8, None)),
                        };
                    } else {
                        return match qrhs {
                            QuantizedTensor::Q8_0(q8) => self.call::<MatMulMlxOp>((
                                a,
                                TensorType::Quant(QuantizedTensor::Q8_0(q8)),
                                None,
                                None,
                                transpose_a,
                                transpose_b,
                                1.0,
                                0.0,
                            )),
                        };
                    }
                }
                let can_gemv = dims.batch == 1 && dims.m == 1 && !transpose_a;
                match qrhs {
                    QuantizedTensor::Q8_0(q8) => {
                        if can_gemv {
                            // Decode-sized matmul: GEMV is best.
                            self.call::<MatmulGemvOp>((a, TensorType::Quant(QuantizedTensor::Q8_0(q8)), None))
                        } else if dims.batch == 1 && (2..=4).contains(&dims.m) && !transpose_a {
                            // For small M (2..=4), canonical kernels beat MLX Q8 across Ns in benches.
                            if self.should_use_q8_canonical_rows16(&dims, transpose_a) {
                                self.call::<MatmulQ8CanonicalRows16Op>((a, q8, None))
                            } else {
                                self.call::<MatmulQ8CanonicalOp>((a, q8, None))
                            }
                        } else if self.should_use_q8_canonical(&dims, transpose_a) {
                            // Large-N canonical kernel is faster than MLX Q8 for small-M in benches.
                            self.call::<MatmulQ8CanonicalOp>((a, q8, None))
                        } else {
                            // Fallback to MLX Q8 loader (tB=false) for other shapes.
                            self.call::<crate::kernels::matmul_mlx::MatMulMlxOp>((
                                a,
                                TensorType::Quant(QuantizedTensor::Q8_0(q8)),
                                None,
                                None,
                                transpose_a,
                                false,
                                1.0,
                                0.0,
                            ))
                        }
                    }
                }
            }
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn fused_qkv_projection(
        &mut self,
        x_flat: &Tensor<T>,
        fused_weight: &Tensor<T>,
        fused_bias: &Tensor<T>,
        d_model: usize,
        kv_dim: usize,
    ) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>), MetalError> {
        return self.with_gpu_scope("attn_qkv_proj".to_string(), |ctx| {
            let x_dims = x_flat.dims();
            if x_dims.len() != 2 {
                return Err(MetalError::InvalidShape(format!(
                    "fused_qkv_projection expects a 2D input [m, d_model], got {:?}",
                    x_dims
                )));
            }

            let _m = x_dims[0];
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

            let linear = ctx.matmul_bias_add(x_flat, &TensorType::Dense(fused_weight), fused_bias, false, false, None)?;

            let q_range_end = d_model;
            let k_range_end = d_model + kv_dim;
            let v_range_end = expected_total;

            let q_out = linear.slice_last_dim(0..q_range_end)?;
            let k_out = linear.slice_last_dim(d_model..k_range_end)?;
            let v_out = linear.slice_last_dim(k_range_end..v_range_end)?;

            Ok((q_out, k_out, v_out))
        });
    }

    /// Fused QKV projection using canonical Q8 tensors for Q, K, V.
    /// Requires m=1 decode (x_flat [1, K]) and equal output widths for Q/K/V.
    pub fn fused_qkv_projection_q8(
        &mut self,
        x_flat: &Tensor<T>,
        wq_q8: &crate::tensor::QuantizedQ8_0Tensor,
        wk_q8: &crate::tensor::QuantizedQ8_0Tensor,
        wv_q8: &crate::tensor::QuantizedQ8_0Tensor,
        q_bias: Tensor<T>,
        k_bias: Tensor<T>,
        v_bias: Tensor<T>,
    ) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>), MetalError> {
        return self.with_gpu_scope("attn_qkv_proj".to_string(), |ctx| {
            if T::DTYPE != crate::tensor::Dtype::F16 {
                return Err(MetalError::UnsupportedDtype {
                    operation: "fused_qkv_projection_q8",
                    dtype: T::DTYPE,
                });
            }
            let x_dims = x_flat.dims();
            if x_dims.len() != 2 || x_dims[0] != 1 {
                return Err(MetalError::InvalidShape(format!(
                    "fused_qkv_projection_q8 expects x_flat=[1,K], got {:?}",
                    x_dims
                )));
            }
            let k = x_dims[1];
            let wq = &wq_q8.logical_dims;
            let wk = &wk_q8.logical_dims;
            let wv = &wv_q8.logical_dims;
            if wq.get(0) != Some(&k) || wk.get(0) != Some(&k) || wv.get(0) != Some(&k) {
                return Err(MetalError::InvalidShape("QKV K mismatch for fused_qkv_projection_q8".into()));
            }
            let nq = *wq.get(1).ok_or(MetalError::InvalidShape("wq dims".into()))?;
            let nk = *wk.get(1).ok_or(MetalError::InvalidShape("wk dims".into()))?;
            let nv = *wv.get(1).ok_or(MetalError::InvalidShape("wv dims".into()))?;

            let y = ctx.call::<MatmulGemvQkvFusedOp>((
                x_flat,
                (
                    &QuantizedTensor::Q8_0(wq_q8),
                    &QuantizedTensor::Q8_0(wk_q8),
                    &QuantizedTensor::Q8_0(wv_q8),
                ),
                (Some(&q_bias), Some(&k_bias), Some(&v_bias)),
            ))?;

            // Build views into the flat fused result: [nq | nk | nv]
            let elem = T::DTYPE.size_bytes();
            let q_out = y.build_view(vec![1, nq], vec![nq, 1], y.offset);
            let k_out = y.build_view(vec![1, nk], vec![nk, 1], y.offset + nq * elem);
            let v_out = y.build_view(vec![1, nv], vec![nv, 1], y.offset + (nq + nk) * elem);

            Ok((q_out, k_out, v_out))
        });
    }

    #[inline]
    pub fn matmul_bias_add(
        &mut self,
        a: &Tensor<T>,
        b_any: &TensorType<T>,
        bias: &Tensor<T>,
        transpose_a: bool,
        transpose_b: bool,
        cache: Option<&mut ResourceCache>,
    ) -> Result<Tensor<T>, MetalError> {
        // TODO: We need to add resource caching
        match b_any {
            TensorType::Dense(bd) => match self.forced_matmul_backend {
                InternalMatMulBackendOverride::Force(backend) => match backend {
                    MatMulBackend::Mlx => {
                        self.call::<MatMulMlxOp>((a, TensorType::Dense(bd), Some(bias), None, transpose_a, transpose_b, 1.0, 0.0))
                    }
                    MatMulBackend::Mps => {
                        let mut linear = self.call_mps_matmul_op((a, bd, transpose_a, transpose_b), cache)?;
                        linear = self.call::<BroadcastElemwiseAddInplaceOp>((linear, bias.clone()))?;
                        Ok(linear)
                    }
                    MatMulBackend::Gemv => self.call::<MatmulGemvOp>((a, TensorType::Dense(bd), Some(bias))),
                },
                InternalMatMulBackendOverride::Default | InternalMatMulBackendOverride::Auto => {
                    let dims_result = self.compute_matmul_dims(a, b_any, transpose_a, transpose_b);

                    match dims_result {
                        Ok(dimensions) => {
                            let use_mlx = self.should_use_mlx_bias(&dimensions);

                            if use_mlx {
                                self.call::<MatMulMlxOp>((a, TensorType::Dense(bd), Some(bias), None, transpose_a, transpose_b, 1.0, 0.0))
                            } else {
                                let mut linear = self.call_mps_matmul_op((a, bd, transpose_a, transpose_b), cache)?;
                                linear = self.call::<BroadcastElemwiseAddInplaceOp>((linear, bias.clone()))?;
                                Ok(linear)
                            }
                        }
                        Err(_) => {
                            let mut linear = self.call_mps_matmul_op((a, bd, transpose_a, transpose_b), cache)?;
                            linear = self.call::<BroadcastElemwiseAddInplaceOp>((linear, bias.clone()))?;
                            Ok(linear)
                        }
                    }
                }
            },
            TensorType::Quant(qrhs) => {
                let dims = self.compute_matmul_dims(a, b_any, transpose_a, transpose_b)?;
                if T::DTYPE != crate::tensor::Dtype::F16 {
                    return Err(MetalError::UnsupportedDtype {
                        operation: "quant matmul_bias_add",
                        dtype: T::DTYPE,
                    });
                }
                if transpose_b {
                    if transpose_a {
                        return Err(MetalError::OperationNotSupported(
                            "quant matmul_bias_add: transpose_a+transpose_b unsupported".into(),
                        ));
                    }
                    // Decode optimization: route tall-N K-small to MLX quant path by using tB=false.
                    if dims.batch == 1 && dims.m == 1 && dims.k <= 1024 && dims.n >= 2048 {
                        return match qrhs {
                            QuantizedTensor::Q8_0(q8) => self.call::<MatMulMlxOp>((
                                a,
                                TensorType::Quant(QuantizedTensor::Q8_0(q8)),
                                Some(bias),
                                None,
                                false, // transpose_right
                                false,
                                1.0,
                                0.0,
                            )),
                        };
                    }
                    // For decode (m=1), GEMV with fused bias is generally faster than NT.
                    if dims.batch == 1 && dims.m == 1 {
                        return match qrhs {
                            QuantizedTensor::Q8_0(q8) => {
                                self.call::<MatmulGemvOp>((a, TensorType::Quant(QuantizedTensor::Q8_0(q8)), Some(bias)))
                            }
                        };
                    }
                    if self.should_use_q8_canonical_rows16(&dims, transpose_a) {
                        return match qrhs {
                            QuantizedTensor::Q8_0(q8) => self.call::<MatmulQ8CanonicalRows16Op>((a, q8, Some(bias))),
                        };
                    }
                    if self.should_use_q8_canonical(&dims, transpose_a) {
                        return match qrhs {
                            QuantizedTensor::Q8_0(q8) => self.call::<MatmulQ8CanonicalOp>((a, q8, Some(bias))),
                        };
                    }
                    let smallm_use_gemv = std::env::var("METALLIC_Q8_SMALLM_USE_GEMV").ok().map(|s| s != "0").unwrap_or(true);
                    let smallm_fallback_max_n = Self::smallm_fallback_max_n();
                    if smallm_use_gemv && dims.batch == 1 && (2..=4).contains(&dims.m) && !transpose_a && dims.n <= smallm_fallback_max_n {
                        return match qrhs {
                            QuantizedTensor::Q8_0(q8) => {
                                self.call::<crate::kernels::matmul_gemv_smallm::MatmulGemvSmallMOp>((a, q8, Some(bias)))
                            }
                        };
                    }
                    if self.can_use_q8_nt(&dims, transpose_a) {
                        return match qrhs {
                            QuantizedTensor::Q8_0(q8) => self.call::<MatmulQ8NtOp>((a, q8, Some(bias))),
                        };
                    } else {
                        return match qrhs {
                            QuantizedTensor::Q8_0(q8) => self.call::<MatMulMlxOp>((
                                a,
                                TensorType::Quant(QuantizedTensor::Q8_0(q8)),
                                Some(bias),
                                None,
                                transpose_a,
                                transpose_b,
                                1.0,
                                0.0,
                            )),
                        };
                    }
                }
                // tB=false branch: prefer GEMV for m=1, otherwise canonical for small-M/large-N, else MLX.
                if dims.batch == 1 && dims.m == 1 && !transpose_a {
                    return match qrhs {
                        QuantizedTensor::Q8_0(q8) => self.call::<MatmulGemvOp>((a, TensorType::Quant(QuantizedTensor::Q8_0(q8)), Some(bias))),
                    };
                }
                if dims.batch == 1 && (2..=4).contains(&dims.m) && !transpose_a {
                    return match qrhs {
                        QuantizedTensor::Q8_0(q8) => {
                            if self.should_use_q8_canonical_rows16(&dims, transpose_a) {
                                self.call::<MatmulQ8CanonicalRows16Op>((a, q8, Some(bias)))
                            } else {
                                self.call::<MatmulQ8CanonicalOp>((a, q8, Some(bias)))
                            }
                        }
                    };
                }
                if self.should_use_q8_canonical(&dims, transpose_a) {
                    return match qrhs {
                        QuantizedTensor::Q8_0(q8) => self.call::<MatmulQ8CanonicalOp>((a, q8, Some(bias))),
                    };
                }
                // Fallback to MLX Q8 bias path (tB=false) for other shapes.
                match qrhs {
                    QuantizedTensor::Q8_0(q8) => self.call::<MatMulMlxOp>((
                        a,
                        TensorType::Quant(QuantizedTensor::Q8_0(q8)),
                        Some(bias),
                        None,
                        transpose_a,
                        false,
                        1.0,
                        0.0,
                    )),
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn matmul_alpha_beta(
        &mut self,
        a: &Tensor<T>,
        b_any: &TensorType<T>,
        result: &Tensor<T>,
        transpose_a: bool,
        transpose_b: bool,
        alpha: f32,
        beta: f32,
        cache: Option<&mut ResourceCache>,
    ) -> Result<Tensor<T>, MetalError> {
        // TODO: add resource cache support for this as well since we dropped the with_cache version
        match b_any {
            TensorType::Dense(bd) => match self.forced_matmul_backend {
                InternalMatMulBackendOverride::Force(backend) => match backend {
                    MatMulBackend::Mlx => {
                        self.call::<MatMulMlxOp>((a, TensorType::Dense(bd), None, Some(result), transpose_a, transpose_b, alpha, beta))
                    }

                    MatMulBackend::Mps => self.call_mps_alpha_beta_op((a, bd, result, transpose_a, transpose_b, alpha, beta), cache),

                    MatMulBackend::Gemv => {
                        self.call::<MatMulMlxOp>((a, TensorType::Dense(bd), None, Some(result), transpose_a, transpose_b, alpha, beta))
                    }
                },
                InternalMatMulBackendOverride::Default | InternalMatMulBackendOverride::Auto => {
                    let dims_result = self.compute_matmul_dims(a, b_any, transpose_a, transpose_b);
                    let (_, use_mlx) = match dims_result {
                        Ok(dimensions) => {
                            let has_strided_batch = self.has_strided_mps_batch(&[a, bd, result]);
                            let use_mlx = self.should_use_mlx_dense(&dimensions, has_strided_batch);
                            (Some(dimensions), use_mlx)
                        }
                        Err(_) => (None, true),
                    };

                    if use_mlx {
                        self.call::<MatMulMlxOp>((a, TensorType::Dense(bd), None, Some(result), transpose_a, transpose_b, alpha, beta))
                    } else {
                        self.call_mps_alpha_beta_op((a, bd, result, transpose_a, transpose_b, alpha, beta), cache)
                    }
                }
            },
            TensorType::Quant(_qrhs) => {
                // TODO add the quant path
                unimplemented!("We havent implemented matmul_bias_add yet")
            }
        }
    }
}
