use crate::metallic::generation::{GenerationConfig, generate};
use crate::metallic::qwen25::{Qwen25, Qwen25Config};
use crate::metallic::{Context, Tokenizer};

#[test]
fn test_generation_pipeline() {
    let mut ctx = Context::new().unwrap();
    let cfg = Qwen25Config {
        n_layers: 1,
        d_model: 8,
        ff_dim: 16,
        n_heads: 2,
        n_kv_heads: 1,
        seq_len: 32,
        vocab_size: 32,
        rope_freq_base: 1e6,
        rms_eps: 1e-6,
    };
    let mut model = Qwen25::new(cfg, &mut ctx).unwrap();

    // Create a simple tokenizer for testing
    use rustc_hash::FxHashMap;
    let mut vocab = FxHashMap::default();
    vocab.insert(0, "<unk>".to_string());
    vocab.insert(1, "hello".to_string());
    vocab.insert(2, "world".to_string());
    vocab.insert(3, "test".to_string());

    // Add more tokens to match the model's vocab_size
    for i in 4..32 {
        vocab.insert(i, format!("token_{}", i));
    }

    let merges = vec![];
    let tokenizer = Tokenizer::from_vocab_and_merges(vocab, merges).unwrap();

    // Test generation with a simple prompt
    let prompt = "hello";
    let gen_config = GenerationConfig {
        max_tokens: 10,
        temperature: 1.0,
        top_p: 0.95,
        top_k: 40,
    };

    // This test just verifies that the generation pipeline can be called
    // without crashing. Actual generation quality would require a trained model.
    let result = generate(&mut model, &tokenizer, &mut ctx, prompt, &gen_config);

    // Print the error if there is one
    match &result {
        Ok(output) => println!("Generation successful: {}", output),
        Err(e) => println!("Generation failed with error: {:?}", e),
    }

    // For this test, we're just checking that the pipeline runs without panicking
    // The actual output will be random since we're using an untrained model
    // We'll consider the test passed if it doesn't panic
    assert!(
        result.is_ok(),
        "Generation should succeed or at least not panic"
    );

    println!("Generation test completed successfully");
}
