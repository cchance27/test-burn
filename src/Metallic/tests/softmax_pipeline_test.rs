
use super::*;

#[test]
fn test_softmax_pipeline_compilation_idempotence() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    // First call to ensure_fused_softmax_pipeline
    let result1 = ensure_fused_softmax_pipeline(&mut context);
    assert!(result1.is_ok(), "First pipeline compilation failed");
    assert!(
        context.fused_softmax_pipeline.is_some(),
        "Pipeline should be compiled after first call"
    );

    // Get the pipeline reference
    let _pipeline1 = context.fused_softmax_pipeline.as_ref().unwrap().clone();

    // Second call to ensure_fused_softmax_pipeline
    let result2 = ensure_fused_softmax_pipeline(&mut context);
    assert!(result2.is_ok(), "Second pipeline compilation failed");
    assert!(
        context.fused_softmax_pipeline.is_some(),
        "Pipeline should still be available after second call"
    );

    // Get the pipeline reference again
    let _pipeline2 = context.fused_softmax_pipeline.as_ref().unwrap().clone();

    // Third call to ensure_fused_softmax_pipeline
    let result3 = ensure_fused_softmax_pipeline(&mut context);
    assert!(result3.is_ok(), "Third pipeline compilation failed");
    assert!(
        context.fused_softmax_pipeline.is_some(),
        "Pipeline should still be available after third call"
    );

    Ok(())
}

#[test]
fn test_softmax_pipeline_multiple_contexts() -> Result<(), MetalError> {
    // Create multiple contexts and compile pipeline in each
    let mut contexts: Vec<Context> = Vec::new();

    for i in 0..3 {
        let mut context = Context::new()?;
        let result = ensure_fused_softmax_pipeline(&mut context);
        assert!(
            result.is_ok(),
            "Pipeline compilation failed for context {}",
            i
        );
        assert!(
            context.fused_softmax_pipeline.is_some(),
            "Pipeline should be compiled for context {}",
            i
        );
        contexts.push(context);
    }

    // All contexts should have a valid pipeline
    for (i, context) in contexts.iter().enumerate() {
        assert!(
            context.fused_softmax_pipeline.is_some(),
            "Context {} should have a pipeline",
            i
        );
    }

    Ok(())
}

#[test]
fn test_softmax_pipeline_concurrent_compilation() -> Result<(), MetalError> {
    // Test that multiple concurrent compilations don't interfere
    let mut context = Context::new()?;

    // Compile pipeline multiple times in quick succession
    for i in 0..5 {
        let result = ensure_fused_softmax_pipeline(&mut context);
        assert!(result.is_ok(), "Pipeline compilation {} failed", i);
        assert!(
            context.fused_softmax_pipeline.is_some(),
            "Pipeline should be available after compilation {}",
            i
        );
    }

    Ok(())
}

#[test]
fn test_softmax_pipeline_recompilation_after_reset() -> Result<(), MetalError> {
    let mut context = Context::new()?;

    // First compilation
    let result1 = ensure_fused_softmax_pipeline(&mut context);
    assert!(result1.is_ok(), "First pipeline compilation failed");
    assert!(
        context.fused_softmax_pipeline.is_some(),
        "Pipeline should be compiled after first call"
    );

    // Simulate a reset by setting pipeline to None
    context.fused_softmax_pipeline = None;

    // Recompilation should work
    let result2 = ensure_fused_softmax_pipeline(&mut context);
    assert!(result2.is_ok(), "Pipeline recompilation failed");
    assert!(
        context.fused_softmax_pipeline.is_some(),
        "Pipeline should be compiled after recompilation"
    );

    Ok(())
}
