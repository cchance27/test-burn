use std::sync::Arc;

use super::{BPETokenizer, BPETokenizerError};
use crate::MetalError;

impl BPETokenizer {
    /// Decode tokens into text.
    pub fn decode(&self, tokens: &[u32]) -> Result<String, MetalError> {
        let text = self.decode_lossless(tokens)?;
        Ok(text.trim().to_string())
    }

    /// Decode tokens while preserving leading and trailing whitespace.
    pub fn decode_lossless(&self, tokens: &[u32]) -> Result<String, MetalError> {
        let mut result = String::new();
        let mut chunk = String::new();
        let mut byte_scratch = Vec::new();

        let start_index = self.start_index(tokens);
        for &token_id in &tokens[start_index..] {
            if let Some(piece) = self.decode_token_arc(token_id, &mut chunk, &mut byte_scratch)? {
                result.push_str(piece.as_ref());
            }
        }

        if !byte_scratch.is_empty() {
            result.push_str(&String::from_utf8_lossy(&byte_scratch));
            byte_scratch.clear();
        }

        Ok(result)
    }

    /// Decode tokens while preserving control/special token text from the vocabulary.
    ///
    /// This is primarily used by chat-template LCP delta logic, where removing control
    /// markers (e.g. `<|im_start|>`) would corrupt continuation prompts and KV history.
    pub fn decode_lossless_preserve_control(&self, tokens: &[u32]) -> Result<String, MetalError> {
        let mut result = String::new();
        let mut chunk = String::new();
        let mut byte_scratch = Vec::new();

        let start_index = self.start_index(tokens);
        for &token_id in &tokens[start_index..] {
            if let Some(piece) = self.decode_token_arc(token_id, &mut chunk, &mut byte_scratch)? {
                result.push_str(piece.as_ref());
                continue;
            }

            if !byte_scratch.is_empty() {
                result.push_str(&String::from_utf8_lossy(&byte_scratch));
                byte_scratch.clear();
            }

            if let Some(token) = self.vocab.get(&token_id) {
                result.push_str(token.as_ref());
            } else if self.special_tokens.eos_token_id != Some(token_id) {
                return Err(BPETokenizerError::InvalidTokenId(token_id).into());
            }
        }

        if !byte_scratch.is_empty() {
            result.push_str(&String::from_utf8_lossy(&byte_scratch));
            byte_scratch.clear();
        }

        Ok(result)
    }

    /// Decode a single token into a shared string. Tokens that can be reused
    /// directly return a clone of the vocabulary `Arc`; tokens that require
    /// byte decoding or post-processing allocate once and move the buffer into
    /// a new `Arc` without additional copies.
    pub fn decode_token_arc(
        &self,
        token_id: u32,
        scratch: &mut String,
        byte_scratch: &mut Vec<u8>,
    ) -> Result<Option<Arc<str>>, MetalError> {
        scratch.clear();

        if self.special_tokens.eos_token_id == Some(token_id) {
            return Ok(None);
        }
        if self.special_tokens.bos_token_id == Some(token_id) || self.special_tokens.pad_token_id == Some(token_id) {
            return Ok(None);
        }

        let token_type = self.token_types.get(&token_id).cloned().unwrap_or(1); // Default to normal
        let token = self.vocab.get(&token_id).ok_or(BPETokenizerError::InvalidTokenId(token_id))?;
        let token_str = token.as_ref();

        // Fast path: pure ASCII tokens can be returned directly (most special/chat tokens).
        if token_type != 6 && token_str.is_ascii() {
            return Ok(Some(Arc::clone(token)));
        }

        match token_type {
            6 => {
                // GGUF byte token representation (rare, but supported).
                if token_str.starts_with("<0x") && token_str.ends_with('>') && token_str.len() == 6 {
                    let byte = u8::from_str_radix(&token_str[3..5], 16).map_err(|_| BPETokenizerError::InvalidTokenId(token_id))?;
                    byte_scratch.push(byte);
                } else {
                    return Err(BPETokenizerError::InvalidTokenId(token_id).into());
                }
            }
            _ => {
                // GPT-2 byte decoder: map token unicode chars back to bytes.
                for ch in token_str.chars() {
                    let idx = ch as usize;
                    if idx < self.byte_decoder_lut.len() {
                        let v = self.byte_decoder_lut[idx];
                        if v != 0 {
                            byte_scratch.push((v - 1) as u8);
                            continue;
                        }
                    }
                    // Fallback: encode the character as UTF-8 bytes.
                    let mut tmp = [0u8; 4];
                    let encoded = ch.encode_utf8(&mut tmp);
                    byte_scratch.extend_from_slice(encoded.as_bytes());
                }
            }
        }

        if byte_scratch.is_empty() {
            return Ok(None);
        }

        // Streaming UTF-8 decode: emit valid prefix, keep incomplete tail (<=3 bytes).
        match std::str::from_utf8(byte_scratch) {
            Ok(s) => {
                scratch.push_str(s);
                byte_scratch.clear();
            }
            Err(err) => {
                let valid = err.valid_up_to();
                if valid > 0 {
                    // SAFETY: `valid_up_to` is always on a UTF-8 boundary.
                    let prefix = std::str::from_utf8(&byte_scratch[..valid]).expect("valid_up_to guaranteed valid UTF-8");
                    scratch.push_str(prefix);
                }

                match err.error_len() {
                    None => {
                        // Incomplete sequence at the end: keep the tail for the next token.
                        let len = byte_scratch.len();
                        if valid < len {
                            byte_scratch.copy_within(valid.., 0);
                            byte_scratch.truncate(len - valid);
                        } else {
                            byte_scratch.clear();
                        }
                    }
                    Some(_) => {
                        // Invalid bytes: fall back to lossy decode and reset.
                        scratch.push_str(&String::from_utf8_lossy(byte_scratch));
                        byte_scratch.clear();
                    }
                }
            }
        }

        if Self::needs_post_process(scratch) {
            self.post_process_in_place(scratch);
        }

        if scratch.is_empty() {
            return Ok(None);
        }
        Ok(Some(Arc::<str>::from(std::mem::take(scratch))))
    }

    fn start_index(&self, tokens: &[u32]) -> usize {
        if self.add_bos_token && self.special_tokens.bos_token_id.is_some() && tokens.first() == self.special_tokens.bos_token_id.as_ref() {
            1
        } else {
            0
        }
    }

    /// Determine whether the provided token requires post-processing.
    fn needs_post_process(token: &str) -> bool {
        token.contains('Ġ') || token.contains('Ċ') || token.contains('đ') || token.contains("  ")
    }

    /// Post-process decoded text to remove BPE artifacts using in-place updates when
    /// no replacements are necessary. This keeps the hot path allocation-free.
    fn post_process_in_place(&self, text: &mut String) {
        if text.is_empty() {
            return;
        }

        let needs_space = text.contains('Ġ');
        let needs_newline = text.contains('Ċ');
        let needs_tab = text.contains('đ');
        let needs_double_space = text.contains("  ");

        if !(needs_space || needs_newline || needs_tab || needs_double_space) {
            return;
        }

        let mut processed = text.clone();
        if needs_space {
            processed = processed.replace('Ġ', " ");
        }
        if needs_newline {
            processed = processed.replace('Ċ', "\n");
        }
        if needs_tab {
            processed = processed.replace('đ', "\t");
        }
        if needs_double_space {
            processed = processed.replace("  ", " ");
        }

        *text = processed;
    }
}
