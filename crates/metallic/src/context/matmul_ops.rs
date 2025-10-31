use super::{main::Context, utils::MatMulBackendOverride as InternalMatMulBackendOverride};
use crate::{
    MetalError, Tensor, caching::ResourceCache, kernels::{
        elemwise_add::BroadcastElemwiseAddInplaceOp, matmul_gemv::MatmulGemvOp, matmul_mlx::MatMulMlxOp, matmul_mps::{MatMulBackend, MatMulMpsAlphaBetaOp, MatMulMpsOp}
    }, tensor::TensorElement
};

#[derive(Debug, Clone, Copy)]
pub struct MatmulDims {
    pub batch: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl<T: TensorElement> Context<T> {
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

    // Compute matmul dims from tensor views
    fn compute_matmul_dims(&self, a: &Tensor<T>, b: &Tensor<T>, transpose_a: bool, transpose_b: bool) -> Result<MatmulDims, MetalError> {
        let a_view = a.as_mps_matrix_batch_view()?;
        let b_view = b.as_mps_matrix_batch_view()?;
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

    #[inline]
    pub fn matmul(&mut self, a: &Tensor<T>, b: &Tensor<T>, transpose_a: bool, transpose_b: bool) -> Result<Tensor<T>, MetalError> {
        match self.forced_matmul_backend {
            InternalMatMulBackendOverride::Force(backend) => match backend {
                MatMulBackend::Mlx => self.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0)),
                MatMulBackend::Mps => self.call::<MatMulMpsOp>((a, b, transpose_a, transpose_b)),
                MatMulBackend::Gemv => {
                    let dims_result = self.compute_matmul_dims(a, b, transpose_a, transpose_b);
                    match dims_result {
                        Ok(dimensions) => {
                            if self.can_use_gemv(&dimensions, transpose_a, transpose_b) {
                                self.call::<MatmulGemvOp>((a, b))
                            } else {
                                self.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0))
                            }
                        }
                        Err(_) => self.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0)),
                    }
                }
            },
            InternalMatMulBackendOverride::Default | InternalMatMulBackendOverride::Auto => {
                let dims_result = self.compute_matmul_dims(a, b, transpose_a, transpose_b);

                match dims_result {
                    Ok(dimensions) => {
                        if self.can_use_gemv(&dimensions, transpose_a, transpose_b) {
                            return self.call::<MatmulGemvOp>((a, b));
                        }

                        let has_strided_batch = self.has_strided_mps_batch(&[a, b]);
                        let use_mlx = self.should_use_mlx_dense(&dimensions, has_strided_batch);

                        if use_mlx {
                            self.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0))
                        } else {
                            self.call::<MatMulMpsOp>((a, b, transpose_a, transpose_b))
                        }
                    }
                    Err(_) => self.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0)),
                }
            }
        }
    }

    #[inline]
    pub(crate) fn matmul_with_cache(
        &mut self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        transpose_a: bool,
        transpose_b: bool,
        cache: &mut ResourceCache,
    ) -> Result<Tensor<T>, MetalError> {
        match self.forced_matmul_backend {
            InternalMatMulBackendOverride::Force(backend) => match backend {
                MatMulBackend::Mlx => self.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0)),
                MatMulBackend::Mps => self.call_with_cache::<MatMulMpsOp>((a, b, transpose_a, transpose_b), cache),
                MatMulBackend::Gemv => {
                    let dims_result = self.compute_matmul_dims(a, b, transpose_a, transpose_b);
                    match dims_result {
                        Ok(dimensions) => {
                            if self.can_use_gemv(&dimensions, transpose_a, transpose_b) {
                                self.call_with_cache::<MatmulGemvOp>((a, b), cache)
                            } else {
                                self.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0))
                            }
                        }
                        Err(_) => self.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0)),
                    }
                }
            },
            InternalMatMulBackendOverride::Default | InternalMatMulBackendOverride::Auto => {
                let dims_result = self.compute_matmul_dims(a, b, transpose_a, transpose_b);

                match dims_result {
                    Ok(dimensions) => {
                        if self.can_use_gemv(&dimensions, transpose_a, transpose_b) {
                            return self.call_with_cache::<MatmulGemvOp>((a, b), cache);
                        }

                        let has_strided_batch = self.has_strided_mps_batch(&[a, b]);
                        let use_mlx = self.should_use_mlx_dense(&dimensions, has_strided_batch);

                        if use_mlx {
                            self.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0))
                        } else {
                            self.call_with_cache::<MatMulMpsOp>((a, b, transpose_a, transpose_b), cache)
                        }
                    }
                    Err(_) => self.call::<MatMulMlxOp>((a, b, None, None, transpose_a, transpose_b, 1.0, 0.0)),
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

        let linear = self.matmul_bias_add(x_flat, fused_weight, fused_bias, false, false)?;

        let q_range_end = d_model;
        let k_range_end = d_model + kv_dim;
        let v_range_end = expected_total;

        let q_out = linear.slice_last_dim(0..q_range_end)?;
        let k_out = linear.slice_last_dim(d_model..k_range_end)?;
        let v_out = linear.slice_last_dim(k_range_end..v_range_end)?;

        Ok((q_out, k_out, v_out))
    }

    #[inline]
    pub fn matmul_bias_add(
        &mut self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        bias: &Tensor<T>,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<Tensor<T>, MetalError> {
        match self.forced_matmul_backend {
            InternalMatMulBackendOverride::Force(backend) => match backend {
                MatMulBackend::Mlx => self.call::<MatMulMlxOp>((a, b, Some(bias), None, transpose_a, transpose_b, 1.0, 0.0)),
                MatMulBackend::Mps => {
                    let mut linear = self.call::<MatMulMpsOp>((a, b, transpose_a, transpose_b))?;
                    linear = self.call::<BroadcastElemwiseAddInplaceOp>((linear, bias.clone()))?;
                    Ok(linear)
                }
                MatMulBackend::Gemv => {
                    let mut linear = self.call::<MatmulGemvOp>((a, b))?;
                    linear = self.call::<BroadcastElemwiseAddInplaceOp>((linear, bias.clone()))?;
                    Ok(linear)
                }
            },
            InternalMatMulBackendOverride::Default | InternalMatMulBackendOverride::Auto => {
                let dims_result = self.compute_matmul_dims(a, b, transpose_a, transpose_b);

                match dims_result {
                    Ok(dimensions) => {
                        let use_mlx = self.should_use_mlx_bias(&dimensions);

                        if use_mlx {
                            self.call::<MatMulMlxOp>((a, b, Some(bias), None, transpose_a, transpose_b, 1.0, 0.0))
                        } else {
                            let mut linear = self.call::<MatMulMpsOp>((a, b, transpose_a, transpose_b))?;
                            linear = self.call::<BroadcastElemwiseAddInplaceOp>((linear, bias.clone()))?;
                            Ok(linear)
                        }
                    }
                    Err(_) => {
                        let mut linear = self.call::<MatMulMpsOp>((a, b, transpose_a, transpose_b))?;
                        linear = self.call::<BroadcastElemwiseAddInplaceOp>((linear, bias.clone()))?;
                        Ok(linear)
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn matmul_alpha_beta(
        &mut self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        result: &Tensor<T>,
        transpose_a: bool,
        transpose_b: bool,
        alpha: f32,
        beta: f32,
    ) -> Result<Tensor<T>, MetalError> {
        match self.forced_matmul_backend {
            InternalMatMulBackendOverride::Force(backend) => match backend {
                MatMulBackend::Mlx => self.call::<MatMulMlxOp>((a, b, None, Some(result), transpose_a, transpose_b, alpha, beta)),

                MatMulBackend::Mps => self.call::<MatMulMpsAlphaBetaOp>((a, b, result, transpose_a, transpose_b, alpha, beta)),

                MatMulBackend::Gemv => self.call::<MatMulMlxOp>((a, b, None, Some(result), transpose_a, transpose_b, alpha, beta)),
            },
            InternalMatMulBackendOverride::Default | InternalMatMulBackendOverride::Auto => {
                let dims_result = self.compute_matmul_dims(a, b, transpose_a, transpose_b);
                let (_, use_mlx) = match dims_result {
                    Ok(dimensions) => {
                        let has_strided_batch = self.has_strided_mps_batch(&[a, b, result]);
                        let use_mlx = self.should_use_mlx_dense(&dimensions, has_strided_batch);
                        (Some(dimensions), use_mlx)
                    }
                    Err(_) => (None, true),
                };

                if use_mlx {
                    self.call::<MatMulMlxOp>((a, b, None, Some(result), transpose_a, transpose_b, alpha, beta))
                } else {
                    self.call::<MatMulMpsAlphaBetaOp>((a, b, result, transpose_a, transpose_b, alpha, beta))
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn matmul_alpha_beta_with_cache(
        &mut self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        result: &Tensor<T>,
        transpose_a: bool,
        transpose_b: bool,
        alpha: f32,
        beta: f32,
        cache: &mut ResourceCache,
    ) -> Result<Tensor<T>, MetalError> {
        match self.forced_matmul_backend {
            InternalMatMulBackendOverride::Force(backend) => match backend {
                MatMulBackend::Mlx => self.call::<MatMulMlxOp>((a, b, None, Some(result), transpose_a, transpose_b, alpha, beta)),
                MatMulBackend::Mps => {
                    self.call_with_cache::<MatMulMpsAlphaBetaOp>((a, b, result, transpose_a, transpose_b, alpha, beta), cache)
                }
                MatMulBackend::Gemv => self.call::<MatMulMlxOp>((a, b, None, Some(result), transpose_a, transpose_b, alpha, beta)),
            },
            InternalMatMulBackendOverride::Default | InternalMatMulBackendOverride::Auto => {
                let dims_result = self.compute_matmul_dims(a, b, transpose_a, transpose_b);
                let (_, use_mlx) = match dims_result {
                    Ok(dimensions) => {
                        let has_strided_batch = self.has_strided_mps_batch(&[a, b, result]);
                        let use_mlx = self.should_use_mlx_dense(&dimensions, has_strided_batch);
                        (Some(dimensions), use_mlx)
                    }
                    Err(_) => (None, true),
                };

                if use_mlx {
                    self.call::<MatMulMlxOp>((a, b, None, Some(result), transpose_a, transpose_b, alpha, beta))
                } else {
                    self.call_with_cache::<MatMulMpsAlphaBetaOp>((a, b, result, transpose_a, transpose_b, alpha, beta), cache)
                }
            }
        }
    }
}
