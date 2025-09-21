use crate::metallic::cache_keys;
use crate::metallic::cache_keys::MpsGemmKey;
use crate::metallic::matmul::MatMulOperation;
use crate::metallic::resource_cache::ResourceCache;
use crate::metallic::{Context, MetalError, Operation, Tensor};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

#[cfg(test)]
mod tests {
    use super::*;

    // CPU-based matrix multiplication for golden testing
    #[allow(clippy::too_many_arguments)]
    fn cpu_matmul(
        a: &[f32],
        a_original_rows: usize,
        a_original_cols: usize,
        b: &[f32],
        b_original_rows: usize,
        b_original_cols: usize,
        transpose_left: bool,
        transpose_right: bool,
    ) -> Vec<f32> {
        let effective_a_rows = if transpose_left {
            a_original_cols
        } else {
            a_original_rows
        };
        let effective_a_cols = if transpose_left {
            a_original_rows
        } else {
            a_original_cols
        };
        let effective_b_rows = if transpose_right {
            b_original_cols
        } else {
            b_original_rows
        };
        let effective_b_cols = if transpose_right {
            b_original_rows
        } else {
            b_original_cols
        };

        assert_eq!(
            effective_a_cols, effective_b_rows,
            "Matrix dimensions are not compatible for multiplication"
        );

        let m = effective_a_rows;
        let n = effective_b_cols;
        let k = effective_a_cols; // or effective_b_rows

        let mut result = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    let a_val = if transpose_left {
                        a[l * a_original_cols + i] // Access a[l][i] when A is transposed (A^T[i,l] = A[l,i])
                    } else {
                        a[i * a_original_cols + l] // Access a[i][l] normally
                    };
                    let b_val = if transpose_right {
                        b[j * b_original_cols + l] // Access b[j][l] when B is transposed (B^T[l,j] = B[j,l])
                    } else {
                        b[l * b_original_cols + j] // Access b[l][j] normally
                    };
                    sum += a_val * b_val;
                }
                result[i * n + j] = sum;
            }
        }
        result
    }

    #[test]
    fn test_matmul_correctness_small_int() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        let mut cache = ResourceCache::new();
        let m = 2; // A rows
        let k = 3; // A cols / B rows
        let n = 2; // B cols

        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let b_data = vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0]; // 3x2 matrix

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![m, k], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![k, n], &context)?;
        let result_tensor = Tensor::zeros(vec![m, n], &mut context)?;

        let cpu_output = cpu_matmul(&a_data, 2, 3, &b_data, 3, 2, false, false);

        let qk_gemm_key = MpsGemmKey {
            transpose_left: false,
            transpose_right: false,
            result_rows: m,
            result_columns: n,
            interior_columns: k,
            alpha: 1.0,
            beta: 0.0,
        };
        let gemm_op = cache.get_or_create_gemm(qk_gemm_key.clone(), &context.device)?;

        let bytes_per_elem: usize = core::mem::size_of::<f32>();

        let a_desc_key = super::cache_keys::MpsMatrixDescriptorKey {
            rows: m,
            columns: k,
            row_bytes: k * bytes_per_elem,
        };
        let b_desc_key = super::cache_keys::MpsMatrixDescriptorKey {
            rows: k,
            columns: n,
            row_bytes: n * bytes_per_elem,
        };
        let result_desc_key = super::cache_keys::MpsMatrixDescriptorKey {
            rows: m,
            columns: n,
            row_bytes: n * bytes_per_elem,
        };

        let matmul_op = MatMulOperation {
            left_buf: a_tensor.buf.clone(),
            left_offset: a_tensor.offset,
            right_buf: b_tensor.buf.clone(),
            right_offset: b_tensor.offset,
            result_buf: result_tensor.buf.clone(),
            result_offset: result_tensor.offset,
            left_desc: cache.get_or_create_descriptor(a_desc_key, &context.device)?,
            right_desc: cache.get_or_create_descriptor(b_desc_key, &context.device)?,
            result_desc: cache.get_or_create_descriptor(result_desc_key, &context.device)?,
            gemm: gemm_op,
        };

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        matmul_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = result_tensor.as_slice();

        assert_eq!(metal_output, cpu_output);

        Ok(())
    }

    #[test]
    fn test_matmul_correctness_asymmetric_float() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        let mut cache = ResourceCache::new();
        let m = 5;
        let k = 4;
        let n = 7;

        let a_data: Vec<f32> = (0..(m * k)).map(|i| (i as f32 * 0.123) - 1.0).collect();
        let b_data: Vec<f32> = (0..(k * n)).map(|i| (i as f32 * 0.456) + 0.5).collect();

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![m, k], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![k, n], &context)?;
        let result_tensor = Tensor::zeros(vec![m, n], &mut context)?;

        let cpu_output = cpu_matmul(&a_data, 5, 4, &b_data, 4, 7, false, false);

        let qk_gemm_key = MpsGemmKey {
            transpose_left: false,
            transpose_right: false,
            result_rows: m,
            result_columns: n,
            interior_columns: k,
            alpha: 1.0,
            beta: 0.0,
        };
        let gemm_op = cache.get_or_create_gemm(qk_gemm_key.clone(), &context.device)?;

        let bytes_per_elem: usize = core::mem::size_of::<f32>();

        let a_desc_key = super::cache_keys::MpsMatrixDescriptorKey {
            rows: m,
            columns: k,
            row_bytes: k * bytes_per_elem,
        };
        let b_desc_key = super::cache_keys::MpsMatrixDescriptorKey {
            rows: k,
            columns: n,
            row_bytes: n * bytes_per_elem,
        };
        let result_desc_key = super::cache_keys::MpsMatrixDescriptorKey {
            rows: m,
            columns: n,
            row_bytes: n * bytes_per_elem,
        };

        let matmul_op = MatMulOperation {
            left_buf: a_tensor.buf.clone(),
            left_offset: a_tensor.offset,
            right_buf: b_tensor.buf.clone(),
            right_offset: b_tensor.offset,
            result_buf: result_tensor.buf.clone(),
            result_offset: result_tensor.offset,
            left_desc: cache.get_or_create_descriptor(a_desc_key, &context.device)?,
            right_desc: cache.get_or_create_descriptor(b_desc_key, &context.device)?,
            result_desc: cache.get_or_create_descriptor(result_desc_key, &context.device)?,
            gemm: gemm_op,
        };

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        matmul_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = result_tensor.as_slice();

        let rtol = 1e-4f64;
        let atol = 1e-6f64;

        for i in 0..(m * n) {
            let metal_val = metal_output[i] as f64;
            let cpu_val = cpu_output[i] as f64;
            let diff = (metal_val - cpu_val).abs();
            let rel_err = if cpu_val.abs() > 1e-8 {
                diff / cpu_val.abs()
            } else {
                diff
            };
            assert!(
                diff <= atol || rel_err <= rtol,
                "Mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
                i,
                metal_val,
                cpu_val,
                diff,
                rel_err
            );
        }

        Ok(())
    }

    #[test]
    fn test_matmul_transpose_right() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        let mut cache = ResourceCache::new();
        let m = 2; // A rows
        let k = 3; // A cols
        let n = 2; // B rows (after transpose)

        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // A: 2x3
        let b_data = vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0]; // B: 2x3 (will be B^T: 3x2)

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![m, k], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![n, k], &context)?; // B is 2x3, but conceptually 3x2 for matmul
        let result_tensor = Tensor::zeros(vec![m, n], &mut context)?;

        let cpu_output = cpu_matmul(&a_data, 2, 3, &b_data, 2, 3, false, true);

        let qk_gemm_key = MpsGemmKey {
            transpose_left: false,
            transpose_right: true,
            result_rows: m,
            result_columns: n,
            interior_columns: k,
            alpha: 1.0,
            beta: 0.0,
        };
        let gemm_op = cache.get_or_create_gemm(qk_gemm_key.clone(), &context.device)?;

        let bytes_per_elem: usize = core::mem::size_of::<f32>();

        let a_desc_key = super::cache_keys::MpsMatrixDescriptorKey {
            rows: m,
            columns: k,
            row_bytes: k * bytes_per_elem,
        };
        let b_desc_key = super::cache_keys::MpsMatrixDescriptorKey {
            rows: n,    // B is 2 rows
            columns: k, // B is 3 cols
            row_bytes: k * bytes_per_elem,
        };
        let result_desc_key = super::cache_keys::MpsMatrixDescriptorKey {
            rows: m,
            columns: n,
            row_bytes: n * bytes_per_elem,
        };

        let matmul_op = MatMulOperation {
            left_buf: a_tensor.buf.clone(),
            left_offset: a_tensor.offset,
            right_buf: b_tensor.buf.clone(),
            right_offset: b_tensor.offset,
            result_buf: result_tensor.buf.clone(),
            result_offset: result_tensor.offset,
            left_desc: cache.get_or_create_descriptor(a_desc_key, &context.device)?,
            right_desc: cache.get_or_create_descriptor(b_desc_key, &context.device)?,
            result_desc: cache.get_or_create_descriptor(result_desc_key, &context.device)?,
            gemm: gemm_op,
        };

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        matmul_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = result_tensor.as_slice();

        let rtol = 1e-4f64;
        let atol = 1e-6f64;

        for i in 0..(m * n) {
            let metal_val = metal_output[i] as f64;
            let cpu_val = cpu_output[i] as f64;
            let diff = (metal_val - cpu_val).abs();
            let rel_err = if cpu_val.abs() > 1e-8 {
                diff / cpu_val.abs()
            } else {
                diff
            };
            assert!(
                diff <= atol || rel_err <= rtol,
                "Mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
                i,
                metal_val,
                cpu_val,
                diff,
                rel_err
            );
        }

        Ok(())
    }

    #[test]
    fn test_matmul_transpose_left() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        let mut cache = ResourceCache::new();
        let m = 3; // A rows (after transpose)
        let k = 2; // A cols
        let n = 3; // B cols

        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // A: 2x3 (will be A^T: 3x2)
        let b_data = vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0]; // B: 2x3

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![k, m], &context)?; // A is 2x3, but conceptually 3x2 for matmul
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![k, n], &context)?;
        let result_tensor = Tensor::zeros(vec![m, n], &mut context)?;

        let cpu_output = cpu_matmul(&a_data, 2, 3, &b_data, 2, 3, true, false);

        let qk_gemm_key = MpsGemmKey {
            transpose_left: true,
            transpose_right: false,
            result_rows: m,
            result_columns: n,
            interior_columns: k,
            alpha: 1.0,
            beta: 0.0,
        };
        let gemm_op = cache.get_or_create_gemm(qk_gemm_key.clone(), &context.device)?;

        let bytes_per_elem: usize = core::mem::size_of::<f32>();

        let a_desc_key = super::cache_keys::MpsMatrixDescriptorKey {
            rows: k,    // A is 2 rows
            columns: m, // A is 3 cols
            row_bytes: m * bytes_per_elem,
        };
        let b_desc_key = super::cache_keys::MpsMatrixDescriptorKey {
            rows: k,
            columns: n,
            row_bytes: n * bytes_per_elem,
        };
        let result_desc_key = super::cache_keys::MpsMatrixDescriptorKey {
            rows: m,
            columns: n,
            row_bytes: n * bytes_per_elem,
        };

        let matmul_op = MatMulOperation {
            left_buf: a_tensor.buf.clone(),
            left_offset: a_tensor.offset,
            right_buf: b_tensor.buf.clone(),
            right_offset: b_tensor.offset,
            result_buf: result_tensor.buf.clone(),
            result_offset: result_tensor.offset,
            left_desc: cache.get_or_create_descriptor(a_desc_key, &context.device)?,
            right_desc: cache.get_or_create_descriptor(b_desc_key, &context.device)?,
            result_desc: cache.get_or_create_descriptor(result_desc_key, &context.device)?,
            gemm: gemm_op,
        };

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        matmul_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = result_tensor.as_slice();

        let rtol = 1e-4f64;
        let atol = 1e-6f64;

        for i in 0..(m * n) {
            let metal_val = metal_output[i] as f64;
            let cpu_val = cpu_output[i] as f64;
            let diff = (metal_val - cpu_val).abs();
            let rel_err = if cpu_val.abs() > 1e-8 {
                diff / cpu_val.abs()
            } else {
                diff
            };
            assert!(
                diff <= atol || rel_err <= rtol,
                "Mismatch at index {}: metal={:.6}, cpu={:.6}, diff={:.2e}, rel={:.2e}",
                i,
                metal_val,
                cpu_val,
                diff,
                rel_err
            );
        }

        Ok(())
    }

    #[test]
    fn test_matmul_alpha_beta_accumulation() -> Result<(), MetalError> {
        let context = Context::new()?;
        let mut cache = ResourceCache::new();
        let m = 2;
        let k = 3;
        let n = 2;

        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let b_data = vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0]; // 3x2 matrix (flattened row-major)
        let c_data = vec![0.5, 1.5, 2.5, 3.5]; // 2x2 matrix

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![m, k], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![k, n], &context)?;
        let c_tensor = Tensor::create_tensor_from_slice(&c_data, vec![m, n], &context)?; // Will be used as result with beta

        let alpha = 0.5;
        let beta = 0.25;

        // Expected result: alpha * A * B + beta * C
        // A * B = [[31, 19], [85, 55]] (calculated with Python)
        // alpha * A * B = [[15.5, 9.5], [42.5, 27.5]]
        // beta * C = [[0.125, 0.375], [0.625, 0.875]]
        // Final result = [[15.625, 9.875], [43.125, 28.375]]
        let expected_result = [15.625, 9.875, 43.125, 28.375];

        let qk_gemm_key = MpsGemmKey {
            transpose_left: false,
            transpose_right: false,
            result_rows: m,
            result_columns: n,
            interior_columns: k,
            alpha,
            beta,
        };
        let gemm_op = cache.get_or_create_gemm(qk_gemm_key.clone(), &context.device)?;

        let bytes_per_elem: usize = core::mem::size_of::<f32>();

        let a_desc_key = super::cache_keys::MpsMatrixDescriptorKey {
            rows: m,
            columns: k,
            row_bytes: k * bytes_per_elem,
        };
        let b_desc_key = super::cache_keys::MpsMatrixDescriptorKey {
            rows: k,
            columns: n,
            row_bytes: n * bytes_per_elem,
        };
        let result_desc_key = super::cache_keys::MpsMatrixDescriptorKey {
            rows: m,
            columns: n,
            row_bytes: n * bytes_per_elem,
        };

        let matmul_op = MatMulOperation {
            left_buf: a_tensor.buf.clone(),
            left_offset: a_tensor.offset,
            right_buf: b_tensor.buf.clone(),
            right_offset: b_tensor.offset,
            result_buf: c_tensor.buf.clone(), // Use C as the result buffer for beta accumulation
            result_offset: c_tensor.offset,
            left_desc: cache.get_or_create_descriptor(a_desc_key, &context.device)?,
            right_desc: cache.get_or_create_descriptor(b_desc_key, &context.device)?,
            result_desc: cache.get_or_create_descriptor(result_desc_key, &context.device)?,
            gemm: gemm_op,
        };

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        matmul_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = c_tensor.as_slice();

        let rtol = 1e-4f64;
        let atol = 1e-6f64;

        for i in 0..(m * n) {
            let metal_val = metal_output[i] as f64;
            let expected_val = expected_result[i];
            let diff = (metal_val - expected_val).abs();
            let rel_err = if expected_val.abs() > 1e-8 {
                diff / expected_val.abs()
            } else {
                diff
            };
            assert!(
                diff <= atol || rel_err <= rtol,
                "Mismatch at index {}: metal={:.6}, expected={:.6}, diff={:.2e}, rel={:.2e}",
                i,
                metal_val,
                expected_val,
                diff,
                rel_err
            );
        }

        Ok(())
    }
}
