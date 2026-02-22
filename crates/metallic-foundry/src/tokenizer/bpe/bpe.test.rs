#![cfg(test)]

use rustc_hash::FxHashMap;

use super::{BPETokenizer, SpecialTokens};

#[test]
fn format_chat_continuation_prompt_inserts_turn_newline_for_chat_templates() {
    // Use a generic `<|im_start|> ... <|im_end|>` template for testing.
    let im_template = "<|im_start|>user\n{{ messages[0]['content'] }}<|im_end|>\n<|im_start|>assistant\n";
    let tokenizer = BPETokenizer::new(
        FxHashMap::default(),
        Vec::new(),
        FxHashMap::default(),
        SpecialTokens::default(),
        false,
        Some(im_template.to_string()),
    )
    .unwrap();

    let formatted = tokenizer.format_chat_continuation_prompt("hello").unwrap();
    assert_eq!(formatted, "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n");
}

#[test]
fn format_chat_continuation_prompt_passthrough_without_chat_template() {
    let tokenizer = BPETokenizer::new(
        FxHashMap::default(),
        Vec::new(),
        FxHashMap::default(),
        SpecialTokens::default(),
        false,
        None,
    )
    .unwrap();

    assert_eq!(tokenizer.format_chat_continuation_prompt("hello").unwrap(), "hello");
}

#[test]
fn decode_maps_gpt2_byte_encoder_space() {
    let mut vocab = FxHashMap::default();
    vocab.insert(0u32, "Ġ".to_string()); // byte-encoded space (0x20)
    vocab.insert(1u32, "h".to_string());

    let tokenizer = BPETokenizer::new(vocab, Vec::new(), FxHashMap::default(), SpecialTokens::default(), false, None).unwrap();

    let decoded = tokenizer.decode_lossless(&[0, 1]).unwrap();
    assert_eq!(decoded, " h");
}

#[test]
fn decode_streaming_joins_multibyte_utf8_sequences() {
    // UTF-8 for "é" is 0xC3 0xA9, which appears as "Ã©" when not byte-decoded.
    let mut vocab = FxHashMap::default();
    vocab.insert(0u32, "Ã".to_string());
    vocab.insert(1u32, "©".to_string());

    let tokenizer = BPETokenizer::new(vocab, Vec::new(), FxHashMap::default(), SpecialTokens::default(), false, None).unwrap();

    let decoded = tokenizer.decode_lossless(&[0, 1]).unwrap();
    assert_eq!(decoded, "é");
}

#[test]
fn decode_lossless_preserve_control_keeps_control_marker_text() {
    let mut vocab = FxHashMap::default();
    vocab.insert(0u32, "<|im_start|>".to_string());
    vocab.insert(1u32, "user".to_string());
    vocab.insert(2u32, "\n".to_string());
    vocab.insert(3u32, "hello".to_string());
    vocab.insert(4u32, "<|im_end|>\n".to_string());

    let mut token_types = FxHashMap::default();
    token_types.insert(0u32, 3); // control
    token_types.insert(4u32, 3); // control

    let tokenizer = BPETokenizer::new(vocab, Vec::new(), token_types, SpecialTokens::default(), false, None).unwrap();

    let tokens = [0u32, 1, 2, 3, 4];
    let plain = tokenizer.decode_lossless(&tokens).unwrap();
    let with_control = tokenizer.decode_lossless_preserve_control(&tokens).unwrap();

    assert_eq!(plain, "<|im_start|>user\nhello<|im_end|>\n");
    assert_eq!(with_control, "<|im_start|>user\nhello<|im_end|>\n");
}

#[test]
fn llama_bpe_pretokenizer_splits_long_digit_runs() {
    let mut vocab = FxHashMap::default();
    vocab.insert(0u32, "?".to_string());
    vocab.insert(1u32, "1234".to_string());
    vocab.insert(2u32, "123".to_string());
    vocab.insert(3u32, "4".to_string());

    let gpt2 = BPETokenizer::new_with_pretokenizer(
        vocab.clone(),
        Vec::new(),
        FxHashMap::default(),
        SpecialTokens::default(),
        false,
        None,
        Some("gpt-2"),
    )
    .unwrap();
    let llama = BPETokenizer::new_with_pretokenizer(
        vocab.clone(),
        Vec::new(),
        FxHashMap::default(),
        SpecialTokens::default(),
        false,
        None,
        Some("llama-bpe"),
    )
    .unwrap();
    let smaug = BPETokenizer::new_with_pretokenizer(
        vocab,
        Vec::new(),
        FxHashMap::default(),
        SpecialTokens::default(),
        false,
        None,
        Some("smaug-bpe"),
    )
    .unwrap();

    assert_eq!(gpt2.encode("1234").unwrap(), vec![1]);
    assert_eq!(llama.encode("1234").unwrap(), vec![2, 3]);
    assert_eq!(smaug.encode("1234").unwrap(), vec![2, 3]);
}

#[test]
fn llama_bpe_pretokenizer_prefers_direct_piece_lookup_before_char_fallback() {
    let mut vocab = FxHashMap::default();
    vocab.insert(0u32, "?".to_string());
    vocab.insert(1u32, "Ġ".to_string());
    vocab.insert(2u32, "a".to_string());
    vocab.insert(3u32, "b".to_string());
    vocab.insert(4u32, "Ġa".to_string());
    vocab.insert(5u32, "Ġab".to_string());

    let merges = vec![("Ġ".to_string(), "a".to_string()), ("Ġa".to_string(), "b".to_string())];

    let gpt2 = BPETokenizer::new_with_pretokenizer(
        vocab.clone(),
        merges.clone(),
        FxHashMap::default(),
        SpecialTokens::default(),
        false,
        None,
        Some("gpt-2"),
    )
    .unwrap();
    let llama = BPETokenizer::new_with_pretokenizer(
        vocab,
        merges,
        FxHashMap::default(),
        SpecialTokens::default(),
        false,
        None,
        Some("llama-bpe"),
    )
    .unwrap();

    assert_eq!(gpt2.encode(" ab").unwrap(), vec![5]);
    assert_eq!(llama.encode(" ab").unwrap(), vec![5]);
}

#[test]
fn llama_bpe_pretokenizer_partially_merges_when_final_piece_missing() {
    let mut vocab = FxHashMap::default();
    vocab.insert(0u32, "?".to_string());
    vocab.insert(1u32, "Ġ".to_string());
    vocab.insert(2u32, "a".to_string());
    vocab.insert(3u32, "b".to_string());
    vocab.insert(4u32, "Ġa".to_string());
    // Intentionally no "Ġab" token.

    let merges = vec![("Ġ".to_string(), "a".to_string()), ("Ġa".to_string(), "b".to_string())];

    let llama = BPETokenizer::new_with_pretokenizer(
        vocab,
        merges,
        FxHashMap::default(),
        SpecialTokens::default(),
        false,
        None,
        Some("llama-bpe"),
    )
    .unwrap();

    assert_eq!(llama.encode(" ab").unwrap(), vec![4, 3]);
}
