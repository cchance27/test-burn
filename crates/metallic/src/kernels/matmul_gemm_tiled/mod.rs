use metallic_instrumentation::GpuProfiler;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputePipelineState, MTLSize};

use crate::{
    CommandBuffer, Context, MetalError, Operation, Tensor, TensorElement, TensorInit, TensorStorage, context::GpuProfilerLabel, encoder::{dispatch_threadgroups, set_buffer, set_bytes, set_compute_pipeline_state}, kernels::{DefaultKernelInvocable, KernelFunction}
};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GemmTiledParams {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub lda: u32,
    pub ldb: u32,
    pub ldc: u32,
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub use_simdgroup_mm: u32,
    pub alpha: f32,
    pub beta: f32,
}

pub struct MatmulGemmTiledOp;

struct MatmulGemmTiled<T: TensorElement> {
    left: Tensor<T>,
    right: Tensor<T>,
    bias: Option<Tensor<T>>,
    out: Tensor<T>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    params: GemmTiledParams,
    threadgroups: MTLSize,
    threads_per_tg: MTLSize,
    profiler_label: GpuProfilerLabel,
}

impl DefaultKernelInvocable for MatmulGemmTiledOp {
    type Args<'a, T: TensorElement> = (
        &'a Tensor<T>,
        &'a Tensor<T>,
        Option<&'a Tensor<T>>,
        Option<&'a Tensor<T>>,
        bool,
        bool,
        f32,
        f32,
    );

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::MatmulGemmTiled)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut crate::caching::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (left, right, bias, existing_out, transpose_left, transpose_right, alpha, beta) = args;

        if beta != 0.0 && existing_out.is_none() {
            return Err(MetalError::InvalidOperation("beta requires an existing output tensor".to_string()));
        }

        let left_view = left.as_mps_matrix_batch_view()?;
        let right_view = right.as_mps_matrix_batch_view()?;

        let batch = left_view.batch;
        if right_view.batch != batch {
            return Err(MetalError::InvalidOperation(
                "Left and right operands must share the same batch".to_string(),
            ));
        }

        let (a_rows_base, a_cols_base) = (left_view.rows, left_view.columns);
        let (b_rows_base, b_cols_base) = (right_view.rows, right_view.columns);

        let (a_rows, a_cols) = if transpose_left {
            (a_cols_base, a_rows_base)
        } else {
            (a_rows_base, a_cols_base)
        };
        let (b_rows, b_cols) = if transpose_right {
            (b_cols_base, b_rows_base)
        } else {
            (b_rows_base, b_cols_base)
        };

        if a_cols != b_rows {
            return Err(MetalError::InvalidOperation(format!(
                "Cannot multiply matrices with shapes {}x{} and {}x{}",
                a_rows, a_cols, b_rows, b_cols
            )));
        }

        let m = a_rows;
        let n = b_cols;
        let k = a_cols;

        if m == 0 || n == 0 || k == 0 {
            return Err(MetalError::InvalidOperation("MatMul dimensions must be non-zero".to_string()));
        }

        let effective_batch = if batch > 1 { batch } else { 1 };

        // Choose tile sizes based on matrix dimensions
        let (tile_m, tile_n, tile_k) = select_tiling_strategy(m, n, k);

        // Determine leading dimensions
        let lda = if transpose_left {
            left_view.matrix_bytes / (T::DTYPE.size_bytes() * a_rows_base)
        } else {
            left_view.matrix_bytes / (T::DTYPE.size_bytes() * a_cols_base)
        } as u32;

        let ldb = if transpose_right {
            right_view.matrix_bytes / (T::DTYPE.size_bytes() * b_rows_base)
        } else {
            right_view.matrix_bytes / (T::DTYPE.size_bytes() * b_cols_base)
        } as u32;

        let out_dims = if effective_batch > 1 {
            vec![effective_batch, m, n]
        } else {
            vec![m, n]
        };
        let out = if let Some(existing) = existing_out {
            existing.clone()
        } else {
            Tensor::new(out_dims, TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?
        };

        let ldc = if effective_batch > 1 {
            // Calculate leading dimension from the tensor's stride information
            // For a 2D matrix, the leading dimension is typically the number of columns for row-major
            // or can be inferred from the stride information
            n as u32 // Use n as leading dimension for the result matrix
        } else {
            n as u32
        };

        // Set up pipeline
        let pipeline = pipeline.unwrap_or_else(|| {
            ctx.kernel_manager
                .get_pipeline(KernelFunction::MatmulGemmTiled, T::DTYPE, &ctx.device)
                .expect("Failed to get matmul_gemm_tiled pipeline")
        });

        // Calculate threadgroup and thread counts
        let tg_count_m = m.div_ceil(tile_m);
        let tg_count_n = n.div_ceil(tile_n);

        let threadgroups = MTLSize {
            width: tg_count_n,
            height: tg_count_m,
            depth: effective_batch,
        };

        // Use appropriate threadgroup size based on tile dimensions to ensure good occupancy
        let threads_per_tg = MTLSize {
            width: 8, // Use 8x8 = 64 threads to match the tile size in the kernel
            height: 8,
            depth: 1,
        };

        let params = GemmTiledParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            lda,
            ldb,
            ldc,
            tile_m: tile_m as u32,
            tile_n: tile_n as u32,
            tile_k: tile_k as u32,
            use_simdgroup_mm: ctx.device_has_simdgroup_mm() as u32,
            alpha,
            beta,
        };

        let tensors_to_prepare = if let Some(bias_tensor) = &bias {
            vec![&left, &right, &out, bias_tensor]
        } else {
            vec![&left, &right, &out]
        };
        ctx.prepare_tensors_for_active_cmd(&tensors_to_prepare)?;

        let profiler_label = ctx
            .take_gpu_scope()
            .unwrap_or_else(|| GpuProfilerLabel::fallback("matmul_gemm_tiled_op"));

        let op = MatmulGemmTiled {
            left: left.clone(),
            right: right.clone(),
            bias: bias.cloned(),
            out: out.clone(),
            pipeline,
            params,
            threadgroups,
            threads_per_tg,
            profiler_label,
        };

        Ok((Box::new(op), out))
    }
}

impl<T: TensorElement> Operation for MatmulGemmTiled<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut crate::caching::ResourceCache) -> Result<(), MetalError> {
        let encoder = command_buffer.get_compute_encoder()?;

        let label = self.profiler_label.clone();
        let _scope = GpuProfiler::profile_compute(command_buffer.raw(), &encoder, label.op_name, label.backend);

        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.left.buf, self.left.offset);
        set_buffer(&encoder, 1, &self.right.buf, self.right.offset);
        if let Some(ref bias) = self.bias {
            set_buffer(&encoder, 2, &self.out.buf, self.out.offset);
            set_bytes(&encoder, 3, &self.params);
            set_buffer(&encoder, 4, &bias.buf, bias.offset);
        } else {
            set_buffer(&encoder, 2, &self.out.buf, self.out.offset);
            set_bytes(&encoder, 3, &self.params);
        }

        dispatch_threadgroups(&encoder, self.threadgroups, self.threads_per_tg);
        Ok(())
    }
}

// Select tiling strategy based on matrix dimensions
// Optimized for modern Apple Silicon (M1/M2/M3) as we only support modern platforms
fn select_tiling_strategy(m: usize, n: usize, k: usize) -> (usize, usize, usize) {
    // Default tile sizes optimized for modern Apple Silicon (M1/M2/M3)
    let (mut tile_m, mut tile_n, mut tile_k) = (32, 32, 16);

    // For very large matrices, use larger tiles to improve cache efficiency and occupancy
    if m >= 1024 || n >= 1024 {
        (tile_m, tile_n, tile_k) = (64, 64, 16);
    }
    // For medium-sized matrices
    else if m >= 256 || n >= 256 {
        (tile_m, tile_n, tile_k) = (64, 32, 16);
    }
    // For skinny matrices (one dimension much smaller), adjust accordingly to maximize parallelism
    else if m == 1 {
        (tile_m, tile_n, tile_k) = (16, 128, 32); // Optimize for row vector by wide matrix
    } else if n == 1 {
        (tile_m, tile_n, tile_k) = (128, 16, 32); // Optimize for tall matrix by column vector
    }
    // For small matrices, use smaller tiles to avoid wasting threads
    else if m < 64 && n < 64 {
        (tile_m, tile_n, tile_k) = (16, 16, 16);
    }

    // Ensure tile sizes are not larger than the maximum supported by the Metal kernel
    // The Metal kernel now uses 8x8 fallback tiles, but allows larger threadgroup tiles
    // The kernel handles tiling internally for large tiles
    let tile_m = tile_m.min(m).clamp(8, 64);
    let tile_n = tile_n.min(n).clamp(8, 64);
    let tile_k = tile_k.min(k).clamp(8, 32);

    (tile_m, tile_n, tile_k)
}

mod matmul_gemm_tiled_test;
#[cfg(test)]
mod tests {
    use super::*;
    // No longer needed

    #[test]
    fn test_select_tiling_strategy() -> Result<(), MetalError> {
        // Test normal case
        let (tm, tn, tk) = select_tiling_strategy(128, 128, 1024);
        assert_eq!((tm, tn, tk), (32, 32, 16));

        // Test large matrices
        let (tm, tn, tk) = select_tiling_strategy(2048, 2048, 4096);
        assert_eq!((tm, tn, tk), (64, 64, 16));

        // Test skinny matrices (but large in other dimensions - gets overridden by large matrix rule)
        let (tm, tn, tk) = select_tiling_strategy(1, 1024, 512);
        assert_eq!((tm, tn, tk), (8, 64, 16)); // Large N overrides skinny M rule

        Ok(())
    }
}
