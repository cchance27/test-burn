use crate::metallic::elemwise_mul::{ElemwiseMul, ensure_mul_pipeline};
use crate::metallic::resource_cache::ResourceCache;
use crate::metallic::{Context, MetalError, Operation, Tensor};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_elemwise_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
    }

    #[test]
    fn test_elemwise_mul_basic() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        ensure_mul_pipeline(&mut context)?;
        let mut cache = ResourceCache::new();

        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![0.5, 1.5, 2.5, 3.5];

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![2, 2], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![2, 2], &context)?;
        let result_tensor = Tensor::zeros(vec![2, 2], &mut context)?;

        let cpu_result = cpu_elemwise_mul(&a_data, &b_data);

        let mul_op = ElemwiseMul::new(
            a_tensor.clone(),
            b_tensor.clone(),
            result_tensor.clone(),
            context.mul_pipeline.as_ref().unwrap().clone(),
        )?;

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        mul_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = result_tensor.as_slice();

        assert_eq!(metal_output, &cpu_result[..]);

        Ok(())
    }

    #[test]
    fn test_elemwise_mul_1d() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        ensure_mul_pipeline(&mut context)?;
        let mut cache = ResourceCache::new();

        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![6], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![6], &context)?;
        let result_tensor = Tensor::zeros(vec![6], &mut context)?;

        let cpu_result = cpu_elemwise_mul(&a_data, &b_data);

        let mul_op = ElemwiseMul::new(
            a_tensor.clone(),
            b_tensor.clone(),
            result_tensor.clone(),
            context.mul_pipeline.as_ref().unwrap().clone(),
        )?;

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        mul_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = result_tensor.as_slice();

        assert_eq!(metal_output, &cpu_result[..]);

        Ok(())
    }

    #[test]
    fn test_elemwise_mul_3d() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        ensure_mul_pipeline(&mut context)?;
        let mut cache = ResourceCache::new();

        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![2, 2, 2], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![2, 2, 2], &context)?;
        let result_tensor = Tensor::zeros(vec![2, 2, 2], &mut context)?;

        let cpu_result = cpu_elemwise_mul(&a_data, &b_data);

        let mul_op = ElemwiseMul::new(
            a_tensor.clone(),
            b_tensor.clone(),
            result_tensor.clone(),
            context.mul_pipeline.as_ref().unwrap().clone(),
        )?;

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        mul_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = result_tensor.as_slice();

        assert_eq!(metal_output, &cpu_result[..]);

        Ok(())
    }

    #[test]
    fn test_elemwise_mul_with_zero() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        ensure_mul_pipeline(&mut context)?;
        let mut cache = ResourceCache::new();

        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![0.0, 0.0, 0.0, 0.0]; // All zeros

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![2, 2], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![2, 2], &context)?;
        let result_tensor = Tensor::zeros(vec![2, 2], &mut context)?;

        let cpu_result = cpu_elemwise_mul(&a_data, &b_data);

        let mul_op = ElemwiseMul::new(
            a_tensor.clone(),
            b_tensor.clone(),
            result_tensor.clone(),
            context.mul_pipeline.as_ref().unwrap().clone(),
        )?;

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        mul_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = result_tensor.as_slice();

        assert_eq!(metal_output, &cpu_result[..]);

        Ok(())
    }

    #[test]
    fn test_elemwise_mul_with_negative_values() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        ensure_mul_pipeline(&mut context)?;
        let mut cache = ResourceCache::new();

        let a_data = vec![-1.0, 2.0, -3.0, 4.0];
        let b_data = vec![0.5, -1.5, 2.5, -3.5];

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![2, 2], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![2, 2], &context)?;
        let result_tensor = Tensor::zeros(vec![2, 2], &mut context)?;

        let cpu_result = cpu_elemwise_mul(&a_data, &b_data);

        let mul_op = ElemwiseMul::new(
            a_tensor.clone(),
            b_tensor.clone(),
            result_tensor.clone(),
            context.mul_pipeline.as_ref().unwrap().clone(),
        )?;

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        mul_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = result_tensor.as_slice();

        assert_eq!(metal_output, &cpu_result[..]);

        Ok(())
    }

    #[test]
    fn test_elemwise_mul_large_tensor() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        ensure_mul_pipeline(&mut context)?;
        let mut cache = ResourceCache::new();

        let size = 1024;
        let a_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.5).collect();
        let b_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.3).collect();

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![32, 32], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![32, 32], &context)?;
        let result_tensor = Tensor::zeros(vec![32, 32], &mut context)?;

        let cpu_result = cpu_elemwise_mul(&a_data, &b_data);

        let mul_op = ElemwiseMul::new(
            a_tensor.clone(),
            b_tensor.clone(),
            result_tensor.clone(),
            context.mul_pipeline.as_ref().unwrap().clone(),
        )?;

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        mul_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = result_tensor.as_slice();

        assert_eq!(metal_output, &cpu_result[..]);

        Ok(())
    }

    #[test]
    fn test_elemwise_mul_floating_precision() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        ensure_mul_pipeline(&mut context)?;
        let mut cache = ResourceCache::new();

        // Using floating point numbers that may introduce precision errors
        let a_data = vec![1.1, 2.2, 3.3, 4.4];
        let b_data = vec![0.1, 0.2, 0.3, 0.4];

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![2, 2], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![2, 2], &context)?;
        let result_tensor = Tensor::zeros(vec![2, 2], &mut context)?;

        let cpu_result = cpu_elemwise_mul(&a_data, &b_data);

        let mul_op = ElemwiseMul::new(
            a_tensor.clone(),
            b_tensor.clone(),
            result_tensor.clone(),
            context.mul_pipeline.as_ref().unwrap().clone(),
        )?;

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        mul_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = result_tensor.as_slice();

        // Using a tolerance for floating point comparison
        for i in 0..metal_output.len() {
            let diff = (metal_output[i] - cpu_result[i]).abs();
            assert!(
                diff < 1e-6,
                "Mismatch at index {}: metal={}, cpu={}, diff={}",
                i,
                metal_output[i],
                cpu_result[i],
                diff
            );
        }

        Ok(())
    }

    #[test]
    fn test_elemwise_mul_numerical_stability() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        ensure_mul_pipeline(&mut context)?;
        let mut cache = ResourceCache::new();

        // Test with large values that could cause overflow
        let a_data = vec![1e5, 1e6, 1e7, -1e5, -1e6, -1e7];
        let b_data = vec![1e5, 1e6, 1e7, -1e5, -1e6, -1e7];

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![2, 3], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![2, 3], &context)?;
        let result_tensor = Tensor::zeros(vec![2, 3], &mut context)?;

        let cpu_result = cpu_elemwise_mul(&a_data, &b_data);

        let mul_op = ElemwiseMul::new(
            a_tensor.clone(),
            b_tensor.clone(),
            result_tensor.clone(),
            context.mul_pipeline.as_ref().unwrap().clone(),
        )?;

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        mul_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = result_tensor.as_slice();

        // Check that the output values are finite (not NaN or infinity)
        for &val in metal_output {
            assert!(
                val.is_finite(),
                "ElemwiseMul output contains non-finite value: {}",
                val
            );
        }

        // Compare with CPU implementation using tolerance
        for i in 0..metal_output.len() {
            let diff = (metal_output[i] - cpu_result[i]).abs();
            let rel_diff = if cpu_result[i].abs() > 1e-8 {
                diff / cpu_result[i].abs()
            } else {
                diff
            };

            // Allow for some floating point precision differences with large numbers
            assert!(
                diff < 1e-3 || rel_diff < 1e-6,
                "Mismatch at index {}: metal={}, cpu={}, diff={}",
                i,
                metal_output[i],
                cpu_result[i],
                diff
            );
        }

        Ok(())
    }
}
