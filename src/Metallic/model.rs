use super::{CommandBuffer, MetalError, Operation, Tensor, resource_cache::ResourceCache};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandQueue;

/// A simple model runner that holds a sequence of operations and runs them in order.
pub struct Model {
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
    pub fn forward(
        &self,
        inputs: &[Tensor],
        command_queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>,
        cache: &mut ResourceCache,
    ) -> Result<Vec<Tensor>, MetalError> {
        if self.operations.is_empty() {
            return Err(MetalError::InvalidShape(
                "Model has no operations".to_string(),
            ));
        }

        // Create a command buffer for this forward pass
        let mut cmd_buf = CommandBuffer::new(command_queue)?;

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

impl Default for Model {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}
