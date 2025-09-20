use super::*;
use crate::metallic::Operation;
use crate::metallic::model::Model;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder as _};

// Simple test operation that just validates it can be called
pub struct TestOperation;

impl Operation for TestOperation {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;
        encoder.endEncoding();
        Ok(())
    }
}

#[test]
fn test_model_basic() -> Result<(), MetalError> {
    let _context = Context::new()?;
    let _cache = ResourceCache::new();

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
    let context = Context::new()?;
    let mut cache = ResourceCache::new();

    let model = Model::new(vec![]);

    // Test empty model
    assert!(model.is_empty());
    assert_eq!(model.len(), 0);

    // Running empty model should return error
    let inputs = vec![];
    let result = model.forward(&inputs, &context.command_queue, &mut cache);
    assert!(result.is_err()); // Should error on empty model

    Ok(())
}

#[test]
fn test_model_multiple_operations() -> Result<(), MetalError> {
    // Create multiple test operations
    let operations: Vec<Box<dyn Operation>> = vec![
        Box::new(TestOperation),
        Box::new(TestOperation),
        Box::new(TestOperation),
    ];

    let model = Model::new(operations);

    assert_eq!(model.len(), 3);
    assert!(!model.is_empty());

    Ok(())
}
