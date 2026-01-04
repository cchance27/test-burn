//! DSL steps for GEMV kernels with parameter resolution.

use serde::{Deserialize, Serialize};

use crate::{
    MetalError, foundry::{
        Foundry, spec::{Ref, Step, TensorBindings}
    }, metals::gemv::{GemvCanonical, GemvColMajor, GemvParams, GemvRowMajor}, types::{KernelArg as _, TensorArg}
};

#[derive(Clone, Copy, Debug)]
enum GemvLayout {
    RowMajor,
    ColMajor,
}

fn resolve_gemv_params(
    mut params: GemvParams,
    layout: GemvLayout,
    matrix: &TensorArg,
    vector_x: &TensorArg,
    result_y: &TensorArg,
) -> Result<GemvParams, MetalError> {
    let dims = matrix.dims();
    if dims.len() != 2 {
        return Err(MetalError::InvalidShape(format!("Gemv matrix must be 2D, got dims={:?}", dims)));
    }

    let (dim_k, dim_n) = match layout {
        GemvLayout::RowMajor => (dims[0], dims[1]),
        GemvLayout::ColMajor => (dims[1], dims[0]),
    };

    let layout_name = match layout {
        GemvLayout::RowMajor => "RowMajor",
        GemvLayout::ColMajor => "ColMajor",
    };

    if params.k == 0 {
        params.k = dim_k as u32;
    } else if params.k as usize != dim_k {
        return Err(MetalError::InvalidShape(format!(
            "Gemv{} expects K={} from matrix dims {:?}, but params.k={}. Check weight layout or switch GEMV op.",
            layout_name, dim_k, dims, params.k
        )));
    }

    if params.n == 0 {
        params.n = dim_n as u32;
    } else if params.n as usize != dim_n {
        return Err(MetalError::InvalidShape(format!(
            "Gemv{} expects N={} from matrix dims {:?}, but params.n={}. Check weight layout or switch GEMV op.",
            layout_name, dim_n, dims, params.n
        )));
    }

    if params.batch == 0 {
        params.batch = 1;
    }

    if params.weights_per_block == 0 {
        params.weights_per_block = 32;
    }
    if params.blocks_per_k == 0 {
        let wpb = params.weights_per_block as usize;
        let blocks = (params.k as usize + wpb - 1) / wpb;
        params.blocks_per_k = blocks as u32;
    }

    if params.stride_x == 0 {
        params.stride_x = if vector_x.dims().len() >= 2 {
            vector_x.strides().get(0).copied().unwrap_or(params.k as usize) as u32
        } else {
            params.k
        };
    }

    if params.stride_y == 0 {
        params.stride_y = if result_y.dims().len() >= 2 {
            result_y.strides().get(0).copied().unwrap_or(params.n as usize) as u32
        } else {
            params.n
        };
    }

    if params.stride_a == 0 {
        params.stride_a = 0;
    }

    if params.stride_w == 0 {
        params.stride_w = match layout {
            GemvLayout::RowMajor => params.n,
            GemvLayout::ColMajor => params.k,
        };
    }

    // stride_scale is unused in the Foundry GEMV kernels; keep zero if unset.
    Ok(params)
}

/// DSL Step for Row-Major GEMV.
#[derive(Debug, Serialize, Deserialize)]
pub struct GemvRowMajorStep {
    pub matrix: Ref,
    pub scale_bytes: Ref,
    pub vector_x: Ref,
    pub result_y: Ref,
    pub params: GemvParams,
    pub bias: Ref,
    pub residual: Ref,
    pub alpha: f32,
    pub beta: f32,
    pub has_bias: u32,
}

#[typetag::serde(name = "GemvRowMajor")]
impl Step for GemvRowMajorStep {
    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        let matrix = bindings.resolve(&self.matrix)?;
        let scale_bytes = bindings.resolve(&self.scale_bytes)?;
        let vector_x = bindings.resolve(&self.vector_x)?;
        let result_y = bindings.resolve(&self.result_y)?;
        let bias = bindings.resolve(&self.bias)?;
        let residual = bindings.resolve(&self.residual)?;

        let params = resolve_gemv_params(self.params, GemvLayout::RowMajor, &matrix, &vector_x, &result_y)?;

        let kernel = GemvRowMajor {
            matrix,
            scale_bytes,
            vector_x,
            result_y,
            params,
            bias,
            residual,
            alpha: self.alpha,
            beta: self.beta,
            has_bias: self.has_bias,
        };

        foundry.run(&kernel)
    }

    fn name(&self) -> &'static str {
        "GemvRowMajor"
    }
}

/// DSL Step for Column-Major GEMV.
#[derive(Debug, Serialize, Deserialize)]
pub struct GemvColMajorStep {
    pub matrix: Ref,
    pub scale_bytes: Ref,
    pub vector_x: Ref,
    pub result_y: Ref,
    pub params: GemvParams,
    pub bias: Ref,
    pub residual: Ref,
    pub alpha: f32,
    pub beta: f32,
    pub has_bias: u32,
}

#[typetag::serde(name = "GemvColMajor")]
impl Step for GemvColMajorStep {
    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        let matrix = bindings.resolve(&self.matrix)?;
        let scale_bytes = bindings.resolve(&self.scale_bytes)?;
        let vector_x = bindings.resolve(&self.vector_x)?;
        let result_y = bindings.resolve(&self.result_y)?;
        let bias = bindings.resolve(&self.bias)?;
        let residual = bindings.resolve(&self.residual)?;

        let params = resolve_gemv_params(self.params, GemvLayout::ColMajor, &matrix, &vector_x, &result_y)?;

        let kernel = GemvColMajor {
            matrix,
            scale_bytes,
            vector_x,
            result_y,
            params,
            bias,
            residual,
            alpha: self.alpha,
            beta: self.beta,
            has_bias: self.has_bias,
        };

        foundry.run(&kernel)
    }

    fn name(&self) -> &'static str {
        "GemvColMajor"
    }
}

/// DSL Step for Canonical GEMV (k-block-major layout).
#[derive(Debug, Serialize, Deserialize)]
pub struct GemvCanonicalStep {
    pub matrix: Ref,
    pub scale_bytes: Ref,
    pub vector_x: Ref,
    pub result_y: Ref,
    pub params: GemvParams,
    pub bias: Ref,
    pub residual: Ref,
    pub alpha: f32,
    pub beta: f32,
    pub has_bias: u32,
}

fn resolve_gemv_canonical_params(
    mut params: GemvParams,
    matrix: &TensorArg,
    vector_x: &TensorArg,
    result_y: &TensorArg,
) -> Result<GemvParams, MetalError> {
    // Canonical layout: matrix is stored as [blocks_per_k * N * weights_per_block] (flat).
    // Prefer inferring K/blocks_per_k from matrix length to avoid mistakes when vector_x is a
    // larger pre-allocated buffer (e.g. attention workspace).
    let matrix_dims = matrix.dims();
    if matrix_dims.is_empty() {
        return Err(MetalError::InvalidShape("GemvCanonical matrix dims must be non-empty".into()));
    }
    let matrix_elems = matrix_dims.iter().product::<usize>();

    if params.weights_per_block == 0 {
        params.weights_per_block = 32;
    }

    // N is the output feature dimension (result_y last dim)
    if params.n == 0 {
        let result_dims = result_y.dims();
        let n = result_dims.last().copied().unwrap_or(0);
        if n == 0 {
            return Err(MetalError::InvalidShape(format!(
                "GemvCanonical result_y must have a non-zero last dim, got dims={:?}",
                result_dims
            )));
        }
        params.n = n as u32;
    }

    // Infer blocks_per_k from canonical weight length when possible.
    if params.blocks_per_k == 0 {
        let n = params.n as usize;
        let wpb = params.weights_per_block as usize;
        let denom = n
            .checked_mul(wpb)
            .ok_or_else(|| MetalError::InvalidShape(format!("GemvCanonical N*weights_per_block overflow: N={n} wpb={wpb}")))?;
        if denom == 0 {
            return Err(MetalError::InvalidShape(format!(
                "GemvCanonical expects N*weights_per_block > 0, got N={} wpb={}",
                n, wpb
            )));
        }
        if matrix_elems % denom != 0 {
            return Err(MetalError::InvalidShape(format!(
                "GemvCanonical weight length {} is not divisible by N*weights_per_block={} (N={}, wpb={}); check canonical swizzle or spec.",
                matrix_elems, denom, n, wpb
            )));
        }
        let blocks = matrix_elems / denom;
        if blocks == 0 {
            return Err(MetalError::InvalidShape(format!(
                "GemvCanonical computed blocks_per_k=0 from matrix_elems={} N={} wpb={}",
                matrix_elems, n, wpb
            )));
        }
        params.blocks_per_k = blocks as u32;
    }

    // K is blocks_per_k * weights_per_block (canonical buffers are padded to block size).
    if params.k == 0 {
        params.k = params.blocks_per_k.checked_mul(params.weights_per_block).ok_or_else(|| {
            MetalError::InvalidShape(format!(
                "GemvCanonical K overflow: blocks_per_k={} weights_per_block={}",
                params.blocks_per_k, params.weights_per_block
            ))
        })?;
    }

    if params.batch == 0 {
        params.batch = 1;
    }

    // Defensive canonical layout validation.
    if matrix_dims.len() == 1 {
        let expected = (params.blocks_per_k as usize)
            .checked_mul(params.n as usize)
            .and_then(|v| v.checked_mul(params.weights_per_block as usize))
            .ok_or_else(|| {
                MetalError::InvalidShape(format!(
                    "GemvCanonical expected_len overflow: blocks_per_k={} N={} wpb={}",
                    params.blocks_per_k, params.n, params.weights_per_block
                ))
            })?;
        if matrix_elems != expected {
            return Err(MetalError::InvalidShape(format!(
                "GemvCanonical weight length mismatch: matrix_elems={} expected={} (blocks_per_k={}, N={}, wpb={}).",
                matrix_elems, expected, params.blocks_per_k, params.n, params.weights_per_block
            )));
        }
    }

    if params.stride_x == 0 {
        params.stride_x = if vector_x.dims().len() >= 2 {
            vector_x.strides().get(0).copied().unwrap_or(params.k as usize) as u32
        } else {
            params.k
        };
    }

    if params.stride_y == 0 {
        params.stride_y = if result_y.dims().len() >= 2 {
            result_y.strides().get(0).copied().unwrap_or(params.n as usize) as u32
        } else {
            params.n
        };
    }

    if params.stride_a == 0 {
        params.stride_a = 0;
    }

    // For canonical layout, stride_w is the stride between k-blocks
    if params.stride_w == 0 {
        params.stride_w = params.n.checked_mul(params.weights_per_block).ok_or_else(|| {
            MetalError::InvalidShape(format!(
                "GemvCanonical stride_w overflow: N={} wpb={}",
                params.n, params.weights_per_block
            ))
        })?;
    }

    // Validate vector_x has enough elements for K (single-batch default).
    let vector_elems = vector_x.dims().iter().product::<usize>();
    if vector_elems < params.k as usize {
        return Err(MetalError::InvalidShape(format!(
            "GemvCanonical vector_x too small: elems={} K={} (dims={:?})",
            vector_elems,
            params.k,
            vector_x.dims()
        )));
    }

    Ok(params)
}

#[typetag::serde(name = "GemvCanonical")]
impl Step for GemvCanonicalStep {
    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        let matrix = bindings.resolve(&self.matrix)?;
        let scale_bytes = bindings.resolve(&self.scale_bytes)?;
        let vector_x = bindings.resolve(&self.vector_x)?;
        let result_y = bindings.resolve(&self.result_y)?;
        let bias = bindings.resolve(&self.bias)?;
        let residual = bindings.resolve(&self.residual)?;

        let params = resolve_gemv_canonical_params(self.params, &matrix, &vector_x, &result_y)?;

        let kernel = GemvCanonical {
            matrix,
            scale_bytes,
            vector_x,
            result_y,
            params,
            bias,
            residual,
            alpha: self.alpha,
            beta: self.beta,
            has_bias: self.has_bias,
        };

        foundry.run(&kernel)
    }

    fn name(&self) -> &'static str {
        "GemvCanonical"
    }
}
