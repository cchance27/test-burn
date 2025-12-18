use crate::{
    GemvError, MetalError, Tensor, TensorElement, tensor::{Dtype, QuantizedQ8_0Tensor, QuantizedTensor, TensorType, quantized::CanonicalQuantTensor}
};

pub const THREADGROUP_WIDTH: usize = 256; // Keep in sync with `kernel.metal`
pub const GEMV_COLS_PER_THREAD: usize = 1; // Increasing past 1 hurt perf in profiling
pub const TILE_N: usize = THREADGROUP_WIDTH * GEMV_COLS_PER_THREAD;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GemvParams {
    pub k: u32,
    pub n: u32,
    pub blocks_per_k: u32,
    pub weights_per_block: u32,
}

#[derive(Clone, Copy)]
pub enum GemvLoaderMode {
    Dense,
    DenseBias,
    Q8Canonical,
    Q8CanonicalBias,
    Q8CanonicalDebug,
}

impl GemvLoaderMode {
    pub fn id(self) -> u32 {
        match self {
            GemvLoaderMode::Dense => 0,
            GemvLoaderMode::DenseBias => 1,
            GemvLoaderMode::Q8Canonical => 2,
            GemvLoaderMode::Q8CanonicalBias => 3,
            GemvLoaderMode::Q8CanonicalDebug => 4,
        }
    }
}

#[derive(Clone)]
pub enum GemvRhsBinding<T: TensorElement> {
    Dense(Tensor<T>),
    QuantCanonical(CanonicalQuantTensor),
}

#[derive(Clone, Copy)]
pub struct QuantMeta<'a> {
    pub source: &'a QuantizedQ8_0Tensor,
    pub blocks_per_k: u32,
    pub weights_per_block: u32,
}

pub struct ResolvedGemvRhs<'a, T: TensorElement> {
    pub binding: GemvRhsBinding<T>,
    pub n: usize,
    pub loader_mode: GemvLoaderMode,
    pub needs_bias_buffer: bool,
    pub quant_meta: Option<QuantMeta<'a>>,
}

#[derive(Clone, Copy)]
pub struct GemvDispatch {
    loader_mode: GemvLoaderMode,
    needs_bias_buffer: bool,
    diag_col: u32,
}

impl GemvDispatch {
    pub fn new(loader_mode: GemvLoaderMode, needs_bias_buffer: bool, diag_col: u32) -> Self {
        Self {
            loader_mode,
            needs_bias_buffer,
            diag_col,
        }
    }

    pub fn loader_id(&self) -> u32 {
        self.loader_mode.id()
    }

    pub fn needs_bias_buffer(&self) -> bool {
        self.needs_bias_buffer
    }

    pub fn diag_col(&self) -> u32 {
        self.diag_col
    }
}

pub fn resolve_rhs<'a, T: TensorElement>(rhs: TensorType<'a, T>, k: usize, has_bias: bool) -> Result<ResolvedGemvRhs<'a, T>, MetalError> {
    match rhs {
        TensorType::Dense(a) => {
            let a_dims = a.dims();
            if a_dims.len() != 2 {
                return Err(GemvError::MatrixShape {
                    expected_k: k,
                    actual: a_dims.to_vec(),
                }
                .into());
            }

            // Check for K in either dimension
            let (is_k, is_n_rows) = if a_dims[0] == k {
                (true, false) // [K, N]
            } else if a_dims[1] == k {
                (false, true) // [N, K]
            } else {
                return Err(GemvError::MatrixShape {
                    expected_k: k,
                    actual: a_dims.to_vec(),
                }
                .into());
            };

            let n = if is_n_rows { a_dims[0] } else { a_dims[1] };

            // To suppress unused warning effectively while keeping logic clear:
            let _ = is_k;

            Ok(ResolvedGemvRhs {
                binding: GemvRhsBinding::Dense(a.clone()),
                n,
                loader_mode: if has_bias {
                    GemvLoaderMode::DenseBias
                } else {
                    GemvLoaderMode::Dense
                },
                needs_bias_buffer: has_bias,
                quant_meta: None,
            })
        }
        TensorType::Quant(qrhs) => {
            if T::DTYPE != Dtype::F16 {
                return Err(MetalError::UnsupportedDtype {
                    operation: "MatmulGemv/Q8",
                    dtype: T::DTYPE,
                });
            }
            match qrhs {
                QuantizedTensor::Q8_0(q8) => {
                    if q8.logical_dims.len() != 2 {
                        return Err(GemvError::MatrixShape {
                            expected_k: k,
                            actual: q8.logical_dims.to_vec(),
                        }
                        .into());
                    }
                    let canonical = CanonicalQuantTensor::from_split_q8_tensor(q8)
                        .map_err(|e| MetalError::InvalidOperation(format!("Failed to canonicalize Q8 tensor: {e}")))?;
                    let d0 = canonical.logical_dims[0];
                    let d1 = canonical.logical_dims[1];
                    let d0_is_k = d0 == k;
                    let d1_is_k = d1 == k;
                    if !d0_is_k && !d1_is_k {
                        return Err(GemvError::QuantShape {
                            expected_k: k,
                            actual: canonical.logical_dims.to_vec(),
                        }
                        .into());
                    }
                    let n = if d0_is_k { d1 } else { d0 };
                    let weights_per_block = canonical.weights_per_block as u32;
                    if weights_per_block == 0 {
                        return Err(GemvError::InvalidQuantParams.into());
                    }
                    let blocks_per_k = k.div_ceil(weights_per_block as usize) as u32;
                    let mut loader_mode = if has_bias {
                        GemvLoaderMode::Q8CanonicalBias
                    } else {
                        GemvLoaderMode::Q8Canonical
                    };
                    if let Ok(col_str) = std::env::var("METALLIC_GEMV_DEBUG_COL") {
                        if col_str.parse::<u32>().is_ok() {
                            loader_mode = GemvLoaderMode::Q8CanonicalDebug;
                        }
                    }
                    Ok(ResolvedGemvRhs {
                        binding: GemvRhsBinding::QuantCanonical(canonical),
                        n,
                        loader_mode,
                        needs_bias_buffer: has_bias,
                        quant_meta: Some(QuantMeta {
                            source: q8,
                            blocks_per_k,
                            weights_per_block,
                        }),
                    })
                }
            }
        }
    }
}
