use crate::metallic::{F32Element, generation::sample_top_k_top_p};

#[test]
fn test_sample_top_k_top_p_extreme_logits() {
    // Test with extremely large logits that could cause overflow in exp function
    let extreme_logits = vec![1000.0f32, -1000.0f32, 500.0f32, -500.0f32];
    let top_k = 4;
    let top_p = 0.95;
    let temperature = 1.0;

    let result = sample_top_k_top_p::<F32Element>(&extreme_logits, top_k, top_p, temperature);

    // Should return a valid token index without panic
    assert!(
        result < extreme_logits.len(),
        "Result index {} is out of bounds for logits length {}",
        result,
        extreme_logits.len()
    );
}

#[test]
fn test_sample_top_k_top_p_extreme_temperature() {
    // Test with extremely small temperature (high scaling factor)
    let logits = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
    let top_k = 4;
    let top_p = 0.95;
    let temperature = 1e-6f32; // Very small temperature

    let result = sample_top_k_top_p::<F32Element>(&logits, top_k, top_p, temperature);

    // Should return a valid token index without panic
    assert!(
        result < logits.len(),
        "Result index {} is out of bounds for logits length {}",
        result,
        logits.len()
    );
}

#[test]
fn test_sample_top_k_top_p_extreme_negative_temperature() {
    // Test with negative temperature (edge case)
    let logits = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
    let top_k = 4;
    let top_p = 0.95;
    let temperature = -1.0f32; // Negative temperature

    let result = sample_top_k_top_p::<F32Element>(&logits, top_k, top_p, temperature);

    // Should return a valid token index without panic
    assert!(
        result < logits.len(),
        "Result index {} is out of bounds for logits length {}",
        result,
        logits.len()
    );
}

#[test]
fn test_sample_top_k_top_p_all_same_logits() {
    // Test with all identical logits
    let logits = vec![5.0f32, 5.0f32, 5.0f32, 5.0f32];
    let top_k = 4;
    let top_p = 0.95;
    let temperature = 1.0;

    let result = sample_top_k_top_p::<F32Element>(&logits, top_k, top_p, temperature);

    // Should return a valid token index without panic
    assert!(
        result < logits.len(),
        "Result index {} is out of bounds for logits length {}",
        result,
        logits.len()
    );
}

#[test]
fn test_sample_top_k_top_p_extremely_large_logits() {
    // Test with extremely large logits to test clamping behavior
    let mut logits = vec![1e10f32; 1000]; // A large vocabulary with all very large values
    logits[0] = 1e20f32; // Make one value extremely large
    logits[1] = 1e19f32; // Make another value very large

    let top_k = 50;
    let top_p = 0.95;
    let temperature = 1.0;

    let result = sample_top_k_top_p::<F32Element>(&logits, top_k, top_p, temperature);

    // Should return a valid token index without panic
    assert!(
        result < logits.len(),
        "Result index {} is out of bounds for logits length {}",
        result,
        logits.len()
    );
}

#[test]
fn test_sample_top_k_top_p_extremely_small_logits() {
    // Test with extremely small (negative) logits
    let mut logits = vec![-1e10f32; 100]; // A vocabulary with all very negative values
    logits[0] = -1e5f32; // Make one slightly less negative

    let top_k = 50;
    let top_p = 0.95;
    let temperature = 1.0;

    let result = sample_top_k_top_p::<F32Element>(&logits, top_k, top_p, temperature);

    // Should return a valid token index without panic
    assert!(
        result < logits.len(),
        "Result index {} is out of bounds for logits length {}",
        result,
        logits.len()
    );
}
