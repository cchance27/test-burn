use objc2_metal_performance_shaders::MPSDataType;

use crate::mps_graph::multi_op::*;

#[test]
fn test_multi_op_graph_builder() {
    // Test that the multi-op builder can be created and basic functionality works

    // Test creating the builder
    let data_type = MPSDataType::Float16;
    let builder = MultiOpGraphBuilder::new(data_type).expect("Failed to create MultiOpGraphBuilder");

    // Verify we can build a fused graph (this might run longer but is important for functionality)
    let executable = builder.build_sdpa_projection(
        1,    // batch
        16,   // seq_q
        16,   // seq_k
        32,   // dim
        32,   // output_dim
        true, // causal
    );

    // The executable should be created successfully
    assert!(executable.is_ok(), "Should be able to create fused SDPA+Projection executable");

    if let Ok(executable) = executable {
        // Test layout extraction
        let feed_layout = extract_feed_layout(&executable).expect("Should extract feed layout");
        let result_layout = extract_result_layout(&executable).expect("Should extract result layout");

        // The fused operation should have more feed bindings than SDPA alone
        assert!(!feed_layout.is_empty(), "Feed layout should not be empty");
        assert_eq!(result_layout.len(), 1, "Should have one result for the projection output");
    }
}

#[test]
fn test_executable_layout_conversion() {
    // Create a builder and executable
    let data_type = MPSDataType::Float16;
    let builder = MultiOpGraphBuilder::new(data_type).expect("Failed to create MultiOpGraphBuilder");

    let executable = builder
        .build_sdpa_projection(
            1,     // batch
            16,    // seq_q
            16,    // seq_k
            32,    // dim
            32,    // output_dim
            false, // causal
        )
        .expect("Failed to build fused SDPA+Projection graph");

    // Test the TryFrom conversion for executable layout
    let layout: ExtendableExecutableLayout = (&executable)
        .try_into()
        .expect("Failed to convert MPSGraphExecutable to ExtendableExecutableLayout");

    assert!(!layout.feed_bindings().is_empty(), "Layout should have feed bindings");
    assert!(!layout.result_bindings().is_empty(), "Layout should have result bindings");
}

#[test]
fn test_executable_layout_trait() {
    // This would require a real MPSGraphExecutable for testing,
    // so we'll focus on ensuring the trait works correctly
    let layout = ExtendableExecutableLayout {
        feed_bindings: vec![GraphFeedBinding::SdpaQuery, GraphFeedBinding::SdpaKey],
        result_bindings: vec![GraphResultBinding::SdpaAttention],
    };

    assert_eq!(layout.feed_bindings().len(), 2);
    assert_eq!(layout.result_bindings().len(), 1);
    assert_eq!(layout.feed_bindings(), &[GraphFeedBinding::SdpaQuery, GraphFeedBinding::SdpaKey]);
    assert_eq!(layout.result_bindings(), &[GraphResultBinding::SdpaAttention]);
}
