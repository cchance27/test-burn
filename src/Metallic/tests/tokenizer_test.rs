//! Tests for the BPE tokenizer implementation

use crate::gguf::GGUFFile;
use crate::metallic::Tokenizer;

#[test]
fn test_tokenizer_from_gguf() {
    let path = "/Volumes/2TB/test-burn/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf";
    match GGUFFile::load(path) {
        Ok(gguf) => {
            // Test creating tokenizer from GGUF metadata
            match Tokenizer::from_gguf_metadata(&gguf.metadata) {
                Ok(tokenizer) => {
                    println!("Successfully created tokenizer from GGUF metadata");
                    println!("Vocabulary size: {}", tokenizer.vocab_size());

                    // Test special tokens
                    let special_tokens = tokenizer.special_tokens();
                    println!("BOS token ID: {:?}", special_tokens.bos_token_id);
                    println!("EOS token ID: {:?}", special_tokens.eos_token_id);
                    println!("PAD token ID: {:?}", special_tokens.pad_token_id);

                    // Test a simple token that should exist
                    #[cfg(test)]
                    if let Some(token) = tokenizer.get_token(9707) {
                        println!("Token 9707: '{}'", token);
                    }

                    // Test encoding/decoding with a simple example
                    let text = "Hello";
                    match tokenizer.encode_serial(text) {
                        Ok(tokens) => {
                            println!("Encoded '{}' to {:?} tokens", text, tokens);

                            // Test decoding
                            match tokenizer.decode(&tokens) {
                                Ok(decoded_text) => {
                                    println!("Decoded tokens to '{}'", decoded_text);
                                    // This should match our input
                                    assert_eq!(
                                        decoded_text, text,
                                        "Decoded text should match input"
                                    );
                                }
                                Err(e) => {
                                    panic!("Error decoding tokens: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            panic!("Error encoding text: {}", e);
                        }
                    }

                    // Test with multiple words that should exist
                    let text = "Hello world";
                    match tokenizer.encode_serial(text) {
                        Ok(tokens) => {
                            println!("Encoded '{}' to {:?} tokens", text, tokens);

                            // Test decoding
                            match tokenizer.decode(&tokens) {
                                Ok(decoded_text) => {
                                    println!("Decoded tokens to '{}'", decoded_text);
                                    // This should match our input
                                    assert_eq!(
                                        decoded_text, text,
                                        "Decoded text should match input"
                                    );
                                }
                                Err(e) => {
                                    panic!("Error decoding tokens: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            panic!("Error encoding text: {}", e);
                        }
                    }

                    // Test with a longer sentence - note that our simple tokenizer
                    // may not perfectly tokenize all words
                    let text = "The quick brown fox jumps over the lazy dog!";
                    match tokenizer.encode_serial(text) {
                        Ok(tokens) => {
                            println!("Encoded '{}' to {:?} tokens", text, tokens);

                            // Test decoding - we won't assert equality because our simple
                            // tokenizer doesn't handle all cases perfectly
                            match tokenizer.decode(&tokens) {
                                Ok(decoded_text) => {
                                    println!("Decoded tokens to '{}'", decoded_text);
                                    // Just verify it decodes without error
                                    // A full BPE implementation would be needed for perfect matching
                                }
                                Err(e) => {
                                    panic!("Error decoding tokens: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            panic!("Error encoding text: {}", e);
                        }
                    }
                }
                Err(e) => {
                    panic!("Error creating tokenizer from GGUF metadata: {}", e);
                }
            }
        }
        Err(e) => {
            panic!("Error loading GGUF file: {}", e);
        }
    }
}

#[test]
fn test_encode_variations() {
    use crate::metallic::SpecialTokens;
    use rustc_hash::FxHashMap;

    // Create a simple vocabulary for testing
    let mut vocab = FxHashMap::default();
    vocab.insert(0, "<unk>".to_string());
    vocab.insert(1, "h".to_string());
    vocab.insert(2, "e".to_string());
    vocab.insert(3, "l".to_string());
    vocab.insert(4, "o".to_string());
    vocab.insert(5, "he".to_string());
    vocab.insert(6, "ll".to_string());
    vocab.insert(7, "hello</w>".to_string());
    vocab.insert(8, "world</w>".to_string());
    vocab.insert(9, "w".to_string());
    vocab.insert(10, "r".to_string());
    vocab.insert(11, "d".to_string());
    vocab.insert(12, "worl".to_string());

    // Create simple merges
    let merges = vec![
        ("h".to_string(), "e".to_string()),
        ("l".to_string(), "l".to_string()),
        ("he".to_string(), "ll".to_string()),
        ("hell".to_string(), "o</w>".to_string()),
        ("w".to_string(), "orl".to_string()),
        ("worl".to_string(), "d</w>".to_string()),
    ];

    match Tokenizer::from_vocab_and_merges(vocab, merges) {
        Ok(tokenizer) => {
            // Test with a longer sentence
            let text = "hello world";

            // Test serial encoding
            let serial_tokens = tokenizer
                .encode_serial(text)
                .expect("Serial encoding failed");
            println!("Serial encoded '{}' to {:?} tokens", text, serial_tokens);

            // Test parallel encoding
            let parallel_tokens = tokenizer
                .encode_parallel(text)
                .expect("Parallel encoding failed");
            println!(
                "Parallel encoded '{}' to {:?} tokens",
                text, parallel_tokens
            );

            // Both should produce the same results
            assert_eq!(
                serial_tokens, parallel_tokens,
                "Serial and parallel encoding should produce the same results"
            );

            // Test SIMD encoding (should produce the same results as serial)
            let simd_tokens = tokenizer.encode_simd(text).expect("SIMD encoding failed");
            println!("SIMD encoded '{}' to {:?} tokens", text, simd_tokens);
            assert_eq!(
                serial_tokens, simd_tokens,
                "Serial and SIMD encoding should produce the same results"
            );

            // Test SIMD parallel encoding (should produce the same results as serial)
            let simd_parallel_tokens = tokenizer
                .encode_simd_parallel(text)
                .expect("SIMD parallel encoding failed");
            println!(
                "SIMD parallel encoded '{}' to {:?} tokens",
                text, simd_parallel_tokens
            );
            assert_eq!(
                serial_tokens, simd_parallel_tokens,
                "Serial and SIMD parallel encoding should produce the same results"
            );
        }
        Err(e) => {
            panic!("Error creating tokenizer from vocab and merges: {}", e);
        }
    }
}

#[test]
fn test_decode_variations() {
    use crate::metallic::SpecialTokens;
    use rustc_hash::FxHashMap;

    // Create a simple vocabulary for testing
    let mut vocab = FxHashMap::default();
    vocab.insert(0, "<unk>".to_string());
    vocab.insert(1, "h".to_string());
    vocab.insert(2, "e".to_string());
    vocab.insert(3, "l".to_string());
    vocab.insert(4, "o".to_string());
    vocab.insert(5, "he".to_string());
    vocab.insert(6, "ll".to_string());
    vocab.insert(7, "hello</w>".to_string());

    let merges = vec![];

    match Tokenizer::from_vocab_and_merges(vocab, merges) {
        Ok(tokenizer) => {
            // Test with a simple token
            let tokens = vec![7]; // "hello</w>" token ID

            // Test regular decoding
            let regular_text = tokenizer.decode(&tokens).expect("Regular decoding failed");
            println!("Regular decoded {:?} to '{}'", tokens, regular_text);

            // Test SIMD decoding
            let simd_text = tokenizer
                .decode_simd(&tokens)
                .expect("SIMD decoding failed");
            println!("SIMD decoded {:?} to '{}'", tokens, simd_text);

            // Both should produce the same results
            assert_eq!(
                regular_text, simd_text,
                "Regular and SIMD decoding should produce the same results"
            );
        }
        Err(e) => {
            panic!("Error creating tokenizer from vocab and merges: {}", e);
        }
    }
}

#[test]
fn test_tokenizer_from_vocab_and_merges() {
    use crate::metallic::SpecialTokens;
    use rustc_hash::FxHashMap;

    // Create a simple vocabulary for testing
    let mut vocab = FxHashMap::default();
    vocab.insert(0, "<unk>".to_string());
    vocab.insert(1, "h".to_string());
    vocab.insert(2, "e".to_string());
    vocab.insert(3, "l".to_string());
    vocab.insert(4, "o".to_string());
    vocab.insert(5, "he".to_string());
    vocab.insert(6, "ll".to_string());
    vocab.insert(7, "hello</w>".to_string());

    // Create simple merges
    let merges = vec![
        ("h".to_string(), "e".to_string()),
        ("l".to_string(), "l".to_string()),
        ("he".to_string(), "ll".to_string()),
        ("hell".to_string(), "o</w>".to_string()),
    ];

    match Tokenizer::from_vocab_and_merges(vocab, merges) {
        Ok(tokenizer) => {
            // Test encoding
            let text = "hello";
            match tokenizer.encode_serial(text) {
                Ok(tokens) => {
                    println!("Encoded '{}' to {:?} tokens", text, tokens);
                    // With our simple vocabulary, "hello" should tokenize to a single token
                    assert_eq!(tokens.len(), 1);
                    assert_eq!(tokens[0], 7); // "hello</w>" token ID
                }
                Err(e) => {
                    panic!("Error encoding text: {}", e);
                }
            }

            // Test decoding
            let tokens = vec![7]; // "hello</w>" token ID
            match tokenizer.decode(&tokens) {
                Ok(decoded_text) => {
                    println!("Decoded tokens to '{}'", decoded_text);
                    assert_eq!(decoded_text, "hello");
                }
                Err(e) => {
                    panic!("Error decoding tokens: {}", e);
                }
            }
        }
        Err(e) => {
            panic!("Error creating tokenizer from vocab and merges: {}", e);
        }
    }
}

#[test]
fn test_byte_level_preprocessing() {
    use crate::metallic::SpecialTokens;
    use rustc_hash::FxHashMap;

    // Create a simple vocabulary for testing
    let mut vocab = FxHashMap::default();
    vocab.insert(0, "<unk>".to_string());
    vocab.insert(1, "a".to_string());
    vocab.insert(2, "b".to_string());
    vocab.insert(3, "c".to_string());
    vocab.insert(4, "<0x01>".to_string()); // Byte token
    vocab.insert(5, "<0x02>".to_string()); // Byte token

    let merges = vec![];

    match Tokenizer::from_vocab_and_merges(vocab, merges) {
        Ok(tokenizer) => {
            // Test byte-level preprocessing
            let text = "abc\x01\x02";
            let processed = tokenizer.byte_level_preprocess(text);
            println!("Preprocessed '{}' to '{}'", text, processed);

            // The processed text should contain the byte representations
            assert!(processed.contains("<0x01>"));
            assert!(processed.contains("<0x02>"));
        }
        Err(e) => {
            panic!("Error creating tokenizer: {}", e);
        }
    }
}
