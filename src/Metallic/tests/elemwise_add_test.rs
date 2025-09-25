use crate::metallic::elemwise_add::{BroadcastElemwiseAdd, ElemwiseAdd, ensure_add_pipeline};
use crate::metallic::resource_cache::ResourceCache;
use crate::metallic::{Context, MetalError, Operation, Tensor};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_elemwise_add(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }

    fn cpu_broadcast_add(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter()
            .enumerate()
            .map(|(i, x)| x + b[i % b.len()])
            .collect()
    }

    #[test]
    fn test_elemwise_add_basic() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        ensure_add_pipeline(&mut context)?;
        let mut cache = ResourceCache::new();

        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![0.5, 1.5, 2.5, 3.5];

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![2, 2], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![2, 2], &context)?;
        let result_tensor = Tensor::zeros(vec![2, 2], &mut context)?;

        let cpu_result = cpu_elemwise_add(&a_data, &b_data);

        let add_op = ElemwiseAdd::new(
            a_tensor.clone(),
            b_tensor.clone(),
            result_tensor.clone(),
            context.add_pipeline.as_ref().unwrap().clone(),
        )?;

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        add_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = result_tensor.as_slice();

        assert_eq!(metal_output, &cpu_result[..]);

        Ok(())
    }

    #[test]
    fn test_elemwise_add_1d() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        ensure_add_pipeline(&mut context)?;
        let mut cache = ResourceCache::new();

        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![6], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![6], &context)?;
        let result_tensor = Tensor::zeros(vec![6], &mut context)?;

        let cpu_result = cpu_elemwise_add(&a_data, &b_data);

        let add_op = ElemwiseAdd::new(
            a_tensor.clone(),
            b_tensor.clone(),
            result_tensor.clone(),
            context.add_pipeline.as_ref().unwrap().clone(),
        )?;

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        add_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = result_tensor.as_slice();

        assert_eq!(metal_output, &cpu_result[..]);

        Ok(())
    }

    #[test]
    fn test_elemwise_add_3d() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        ensure_add_pipeline(&mut context)?;
        let mut cache = ResourceCache::new();

        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![2, 2, 2], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![2, 2, 2], &context)?;
        let result_tensor = Tensor::zeros(vec![2, 2, 2], &mut context)?;

        let cpu_result = cpu_elemwise_add(&a_data, &b_data);

        let add_op = ElemwiseAdd::new(
            a_tensor.clone(),
            b_tensor.clone(),
            result_tensor.clone(),
            context.add_pipeline.as_ref().unwrap().clone(),
        )?;

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        add_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = result_tensor.as_slice();

        assert_eq!(metal_output, &cpu_result[..]);

        Ok(())
    }

    #[test]
    fn test_broadcast_add_1d_bias() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        ensure_add_pipeline(&mut context)?;
        let mut cache = ResourceCache::new();

        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // Shape [2, 3]
        let b_data = vec![0.5, 1.0, 1.5]; // Shape [3] - broadcast along last dimension

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![2, 3], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![3], &context)?;
        let result_tensor = Tensor::zeros(vec![2, 3], &mut context)?;

        let cpu_result = cpu_broadcast_add(&a_data, &b_data);

        let broadcast_add_op = BroadcastElemwiseAdd::new(
            a_tensor.clone(),
            b_tensor.clone(),
            result_tensor.clone(),
            context.broadcast_add_pipeline.as_ref().unwrap().clone(),
        )?;

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        broadcast_add_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = result_tensor.as_slice();

        assert_eq!(metal_output, &cpu_result[..]);

        Ok(())
    }

    #[test]
    fn test_broadcast_add_2d_bias() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        ensure_add_pipeline(&mut context)?;
        let mut cache = ResourceCache::new();

        let a_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]; // Shape [2, 2, 3]
        let b_data = vec![0.5, 1.0, 1.5]; // Shape [3] - broadcast along last dimension

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![2, 2, 3], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![3], &context)?;
        let result_tensor = Tensor::zeros(vec![2, 2, 3], &mut context)?;

        let cpu_result = cpu_broadcast_add(&a_data, &b_data);

        let broadcast_add_op = BroadcastElemwiseAdd::new(
            a_tensor.clone(),
            b_tensor.clone(),
            result_tensor.clone(),
            context.broadcast_add_pipeline.as_ref().unwrap().clone(),
        )?;

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        broadcast_add_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = result_tensor.as_slice();

        assert_eq!(metal_output, &cpu_result[..]);

        Ok(())
    }

    #[test]
    fn test_broadcast_add_singleton_broadcast() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        ensure_add_pipeline(&mut context)?;
        let mut cache = ResourceCache::new();

        let a_data = vec![1.0, 2.0, 3.0, 4.0]; // Shape [4]
        let b_data = vec![2.5]; // Shape [1] - broadcast to all elements

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![4], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![1], &context)?;
        let result_tensor = Tensor::zeros(vec![4], &mut context)?;

        let cpu_result = cpu_broadcast_add(&a_data, &b_data);

        let broadcast_add_op = BroadcastElemwiseAdd::new(
            a_tensor.clone(),
            b_tensor.clone(),
            result_tensor.clone(),
            context.broadcast_add_pipeline.as_ref().unwrap().clone(),
        )?;

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        broadcast_add_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = result_tensor.as_slice();

        assert_eq!(metal_output, &cpu_result[..]);

        Ok(())
    }

    #[test]
    fn test_elemwise_add_large_tensor() -> Result<(), MetalError> {
        let mut context = Context::new()?;
        ensure_add_pipeline(&mut context)?;
        let mut cache = ResourceCache::new();

        let size = 1024;
        let a_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.5).collect();
        let b_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.3).collect();

        let a_tensor = Tensor::create_tensor_from_slice(&a_data, vec![32, 32], &context)?;
        let b_tensor = Tensor::create_tensor_from_slice(&b_data, vec![32, 32], &context)?;
        let result_tensor = Tensor::zeros(vec![32, 32], &mut context)?;

        let cpu_result = cpu_elemwise_add(&a_data, &b_data);

        let add_op = ElemwiseAdd::new(
            a_tensor.clone(),
            b_tensor.clone(),
            result_tensor.clone(),
            context.add_pipeline.as_ref().unwrap().clone(),
        )?;

        let command_buffer = context.command_queue.commandBuffer().unwrap();
        add_op.encode(&command_buffer, &mut cache)?;
        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        let metal_output = result_tensor.as_slice();

        assert_eq!(metal_output, &cpu_result[..]);

        Ok(())
    }
}
