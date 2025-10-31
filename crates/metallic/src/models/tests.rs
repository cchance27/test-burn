#![cfg(test)]
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::{CommandBuffer, Context, F32Element, MetalError, Operation, Tensor, TensorElement, caching::ResourceCache};

// Simple test operation that just validates it can be called
pub struct TestOperation;

impl Operation for TestOperation {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        let _encoder = command_buffer.get_compute_encoder()?;
        Ok(())
    }
    fn bind_to_encoder(&self, _encoder: &Retained<ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>>) {
        // No-op for test operation
    }
}

#[test]
fn test_model_basic() -> Result<(), MetalError> {
    let context: Context<F32Element> = Context::new()?;
    let _cache = ResourceCache::with_device(context.device.clone());

    // Create a simple test operation
    let test_op = TestOperation;

    // Create model and add operation
    let mut model = Model::new(vec![]);
    model.add_operation(Box::new(test_op));

    // Test basic properties
    assert_eq!(model.len(), 1);
    assert!(!model.is_empty());

    Ok(())
}

#[test]
fn test_model_empty() -> Result<(), MetalError> {
    let context: Context<F32Element> = Context::new()?;
    let mut cache = ResourceCache::with_device(context.device.clone());

    let model = Model::new(vec![]);

    // Test empty model
    assert!(model.is_empty());
    assert_eq!(model.len(), 0);

    // Running empty model should return error
    let inputs: Vec<Tensor<F32Element>> = vec![];
    let result = model.forward(&inputs, &context.command_queue, &mut cache);
    assert!(result.is_err()); // Should error on empty model

    Ok(())
}

#[test]
fn test_model_multiple_operations() -> Result<(), MetalError> {
    // Create multiple test operations
    let operations: Vec<Box<dyn Operation>> = vec![Box::new(TestOperation), Box::new(TestOperation), Box::new(TestOperation)];

    let model = Model::new(operations);

    assert_eq!(model.len(), 3);
    assert!(!model.is_empty());

    Ok(())
}

/// A simple model runner that holds a sequence of operations and runs them in order.
struct Model {
    operations: Vec<Box<dyn Operation>>,
}

impl Model {
    /// Create a new model with the given operations.
    pub fn new(operations: Vec<Box<dyn Operation>>) -> Self {
        Self { operations }
    }

    /// Add an operation to the model.
    pub fn add_operation(&mut self, operation: Box<dyn Operation>) {
        self.operations.push(operation);
    }

    /// Run the model forward with the given input tensors.
    /// Returns the output tensors from the final operation.
    pub fn forward<T: TensorElement>(
        &self,
        inputs: &[Tensor<T>],
        command_queue: &Retained<ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
        cache: &mut ResourceCache,
    ) -> Result<Vec<Tensor<T>>, MetalError> {
        if self.operations.is_empty() {
            return Err(MetalError::InvalidShape("Model has no operations".to_string()));
        }

        // Create a command buffer for this forward pass
        let cmd_buf = crate::CommandBuffer::new(command_queue)?;

        // For now, we assume the first operation takes the inputs directly
        // In a more sophisticated implementation, we might need to handle
        // intermediate tensor passing between operations
        let current_inputs = inputs.to_vec();

        // Execute all operations in sequence
        for operation in &self.operations {
            cmd_buf.record(operation.as_ref(), cache)?;
        }

        // Commit and wait for completion
        cmd_buf.commit();
        cmd_buf.wait();

        // For now, return the original inputs as outputs
        // In a real implementation, we'd track intermediate outputs
        Ok(current_inputs)
    }

    /// Get the number of operations in the model.
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Check if the model is empty.
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
}
