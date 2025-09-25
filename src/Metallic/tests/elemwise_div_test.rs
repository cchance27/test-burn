use crate::metallic::elemwise_div::{ElemwiseDiv, ensure_div_pipeline};
use crate::metallic::resource_cache::ResourceCache;
use crate::metallic::{Context, MetalError, Operation, Tensor};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_elemwise_div(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x / y).collect()
    }

    #[test]
    fn test_elemwise_div_basic() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        ensure_div_pipeline(&mut context)?;
        let mut cache = ResourceCache::new();

        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![0.5, 1.0, 1.5, 2.0];

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![2, 2], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![2, 2], &context)?;
        let result_tensor = Tensor::zeros(vec![2, 2], &mut context)?;

        let cpu_result = cpu_elemwise_div(&a_data, &b_data);

        let div_op = ElemwiseDiv::new(
            a_tensor.clone(),
            b_tensor.clone(),
            result_tensor.clone(),
            context.div_pipeline.as_ref().unwrap().clone(),
        )?;

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        div_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = result_tensor.as_slice();

        assert_eq!(metal_output, &cpu_result[..]);

        Ok(())
    }

    #[test]
    fn test_elemwise_div_1d() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        ensure_div_pipeline(&mut context)?;
        let mut cache = ResourceCache::new();

        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![6], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![6], &context)?;
        let result_tensor = Tensor::zeros(vec![6], &mut context)?;

        let cpu_result = cpu_elemwise_div(&a_data, &b_data);

        let div_op = ElemwiseDiv::new(
            a_tensor.clone(),
            b_tensor.clone(),
            result_tensor.clone(),
            context.div_pipeline.as_ref().unwrap().clone(),
        )?;

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        div_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = result_tensor.as_slice();

        assert_eq!(metal_output, &cpu_result[..]);

        Ok(())
    }

    #[test]
    fn test_elemwise_div_3d() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        ensure_div_pipeline(&mut context)?;
        let mut cache = ResourceCache::new();

        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![2, 2, 2], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![2, 2, 2], &context)?;
        let result_tensor = Tensor::zeros(vec![2, 2, 2], &mut context)?;

        let cpu_result = cpu_elemwise_div(&a_data, &b_data);

        let div_op = ElemwiseDiv::new(
            a_tensor.clone(),
            b_tensor.clone(),
            result_tensor.clone(),
            context.div_pipeline.as_ref().unwrap().clone(),
        )?;

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        div_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = result_tensor.as_slice();

        assert_eq!(metal_output, &cpu_result[..]);

        Ok(())
    }

    #[test]
    fn test_elemwise_div_by_one() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        ensure_div_pipeline(&mut context)?;
        let mut cache = ResourceCache::new();

        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![1.0, 1.0, 1.0, 1.0]; // Dividing by 1 should leave values unchanged

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![2, 2], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![2, 2], &context)?;
        let result_tensor = Tensor::zeros(vec![2, 2], &mut context)?;

        let cpu_result = cpu_elemwise_div(&a_data, &b_data);

        let div_op = ElemwiseDiv::new(
            a_tensor.clone(),
            b_tensor.clone(),
            result_tensor.clone(),
            context.div_pipeline.as_ref().unwrap().clone(),
        )?;

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        div_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = result_tensor.as_slice();

        assert_eq!(metal_output, &cpu_result[..]);

        Ok(())
    }

    #[test]
    fn test_elemwise_div_with_fractions() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        ensure_div_pipeline(&mut context)?;
        let mut cache = ResourceCache::new();

        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![2.0, 4.0, 6.0, 8.0];

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![2, 2], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![2, 2], &context)?;
        let result_tensor = Tensor::zeros(vec![2, 2], &mut context)?;

        let cpu_result = cpu_elemwise_div(&a_data, &b_data);

        let div_op = ElemwiseDiv::new(
            a_tensor.clone(),
            b_tensor.clone(),
            result_tensor.clone(),
            context.div_pipeline.as_ref().unwrap().clone(),
        )?;

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        div_op.encode(&command_buffer, &mut cache)?;
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
    fn test_elemwise_div_large_tensor() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        ensure_div_pipeline(&mut context)?;
        let mut cache = ResourceCache::new();

        let size = 1024;
        let a_data: Vec<f32> = (1..=size).map(|i| i as f32 * 0.5).collect();
        let b_data: Vec<f32> = (1..=size).map(|i| i as f32 * 0.3).collect();

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![32, 32], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![32, 32], &context)?;
        let result_tensor = Tensor::zeros(vec![32, 32], &mut context)?;

        let cpu_result = cpu_elemwise_div(&a_data, &b_data);

        let div_op = ElemwiseDiv::new(
            a_tensor.clone(),
            b_tensor.clone(),
            result_tensor.clone(),
            context.div_pipeline.as_ref().unwrap().clone(),
        )?;

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        div_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = result_tensor.as_slice();

        // Using a tolerance for floating point comparison due to potential precision issues
        assert_eq!(metal_output.len(), cpu_result.len());
        for i in 0..metal_output.len() {
            let diff = (metal_output[i] - cpu_result[i]).abs();
            assert!(
                diff < 1e-5,
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
    fn test_elemwise_div_floating_precision() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        ensure_div_pipeline(&mut context)?;
        let mut cache = ResourceCache::new();

        // Using floating point numbers that may introduce precision errors
        let a_data = vec![1.1, 2.2, 3.3, 4.4];
        let b_data = vec![0.1, 0.2, 0.3, 0.4];

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![2, 2], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![2, 2], &context)?;
        let result_tensor = Tensor::zeros(vec![2, 2], &mut context)?;

        let cpu_result = cpu_elemwise_div(&a_data, &b_data);

        let div_op = ElemwiseDiv::new(
            a_tensor.clone(),
            b_tensor.clone(),
            result_tensor.clone(),
            context.div_pipeline.as_ref().unwrap().clone(),
        )?;

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        div_op.encode(&command_buffer, &mut cache)?;
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
}
