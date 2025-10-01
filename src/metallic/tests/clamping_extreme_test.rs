use crate::metallic::{F32Element, generation::sample_top_k_top_p};

#[test]
fn test_sample_top_k_top_p_extreme_logits() {
    // Test with extremely large logits that could cause overflow in exp function
    let extreme_logits = vec![1000.0f32, -1000.0f32, 500.0f32, -500.0f32];
    let top_k = 4;
    let top_p = 0.95;
    let temperature = 1.0;

    let result = sample_top_k_top_p::<F32Element>(&extreme_logits, top_k, top_p, temperature);

    assert_eq!(result, 0, "Highest logit should be selected even with extreme values");
}

#[test]
fn test_sample_top_k_top_p_extreme_temperature() {
    // Test with extremely small temperature (high scaling factor)
    let logits = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
    let top_k = 4;
    let top_p = 0.95;
    let temperature = 1e-6f32; // Very small temperature

    let result = sample_top_k_top_p::<F32Element>(&logits, top_k, top_p, temperature);

    assert_eq!(result, 3, "Extremely low temperature should fall back to the maximum logit");
}

#[test]
fn test_sample_top_k_top_p_extreme_negative_temperature() {
    // Test with negative temperature (edge case)
    let logits = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
    let top_k = 4;
    let top_p = 0.95;
    let temperature = -1.0f32; // Negative temperature

    let result = sample_top_k_top_p::<F32Element>(&logits, top_k, top_p, temperature);

    assert_eq!(
        result, 3,
        "Negative temperature should trigger greedy fallback to the maximum logit"
    );
}

#[test]
fn test_sample_top_k_top_p_all_same_logits() {
    // Test with all identical logits
    let logits = vec![5.0f32, 5.0f32, 5.0f32, 5.0f32];
    let top_k = 4;
    let top_p = 0.95;
    let temperature = 0.0; // Greedy path should select the last max index

    let result = sample_top_k_top_p::<F32Element>(&logits, top_k, top_p, temperature);

    assert_eq!(result, 3, "Zero temperature should return the last index with the maximum logit");
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

    assert_eq!(result, 0, "The dominant logit should be selected after clamping large values");
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

    assert_eq!(result, 0, "The least negative logit should be selected after clamping small values");
}

#[test]
fn test_sample_top_k_top_p_with_nan_logits() {
    let logits = vec![f32::NAN, -2.0f32, f32::NAN, 1.0f32, 0.5f32];
    let top_k = 5;
    let top_p = 0.4;
    let temperature = 1.0f32;

    let result = sample_top_k_top_p::<F32Element>(&logits, top_k, top_p, temperature);

    assert_eq!(result, 3, "NaN logits should be skipped while sampling");
}
