// Tokenizer implementation for the Metallic framework.
//!
//! This module provides BPE (Byte Pair Encoding) tokenization capabilities.

use std::sync::{Arc, RwLock};

use fancy_regex::Regex;
use rustc_hash::FxHashMap;
use thiserror::Error;

use crate::{MetalError, template::ChatTemplate};

mod bytes;
mod chat;
mod decode;
mod encode;
mod init;
mod pretokenizer;

use pretokenizer::PreTokenizerKind;

/// Error types for tokenizer operations.
#[derive(Debug, Error)]
pub enum BPETokenizerError {
    #[error("Invalid token ID: {0}")]
    InvalidTokenId(u32),
    #[error("Invalid UTF-8 sequence")]
    InvalidUtf8,
    #[error("Missing vocabulary or merges data")]
    MissingData,
    #[error("Tokenizer initialization failed: {0}")]
    InitializationFailed(String),
    #[error("Regex BPETokenizer Errors: {0}")]
    RegexError(#[from] fancy_regex::Error),
}

/// Special token IDs.
#[derive(Default, Debug, Clone)]
pub struct SpecialTokens {
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
}

/// A BPE tokenizer implementation.
pub struct BPETokenizer {
    /// Vocabulary mapping token IDs to token strings.
    vocab: FxHashMap<u32, Arc<str>>,
    /// Reverse vocabulary mapping token strings to token IDs.
    vocab_r: FxHashMap<String, u32>,
    /// BPE merges.
    merges: FxHashMap<(String, String), u32>,
    /// Token types.
    token_types: FxHashMap<u32, i32>,
    /// Special tokens.
    special_tokens: SpecialTokens,
    /// Whether to add BOS token.
    add_bos_token: bool,
    /// Dynamic chat template for formatting multi-turn prompts.
    chat_template: Option<ChatTemplate>,
    /// Cache for BPE tokenization results.
    bpe_cache: RwLock<FxHashMap<String, Vec<u32>>>,
    /// ID-based BPE merges for optimized tokenization.
    merges_ranks: FxHashMap<(u32, u32), u32>,
    /// ID-based BPE merge results for optimized tokenization.
    merges_results: FxHashMap<(u32, u32), u32>,
    /// Array for O(1) byte-to-unicode mapping.
    byte_encoder_array: [char; 256],
    /// LUT for inverse byte decoding (unicode codepoint -> byte+1).
    ///
    /// The GPT-2 byte encoder maps all 256 bytes into the range U+0021..U+00FF and U+0100..U+01FF,
    /// so a small fixed LUT is enough for the hot path.
    byte_decoder_lut: [u16; 512],
    /// Cache for single-character token lookups.
    char_vocab: FxHashMap<char, u32>,
    /// Precompiled regex for special token spans (`<|...|>`).
    special_token_re: Regex,
    /// Literal control/special tokens that should be matched as atomic pieces.
    special_literal_tokens: Vec<Arc<str>>,
    /// Precompiled regex for normal tokenization pieces.
    token_piece_re: Regex,
    /// Pre-tokenizer family selected from metadata.
    pre_tokenizer_kind: PreTokenizerKind,
    /// Some GGUF pre-tokenizer variants may require bypassing merge ranks.
    /// Currently disabled for all known variants.
    ignore_merges: bool,
}

impl BPETokenizer {
    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get special tokens.
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    pub fn chat_template(&self) -> Option<&ChatTemplate> {
        self.chat_template.as_ref()
    }

    /// Returns true if the tokenizer vocabulary contains the exact token string.
    #[inline]
    pub fn has_token(&self, token: &str) -> bool {
        self.vocab_r.contains_key(token)
    }

    /// Get a token by ID (for testing purposes).
    #[cfg(test)]
    pub fn get_token(&self, id: u32) -> Option<Arc<str>> {
        self.vocab.get(&id).cloned()
    }

    /// Get a token by ID (for debugging purposes).
    pub fn get_token_debug(&self, id: u32) -> Option<Arc<str>> {
        self.vocab.get(&id).cloned()
    }

    /// Clear the BPE cache.
    pub fn clear_cache(&self) -> Result<(), MetalError> {
        let mut cache = self
            .bpe_cache
            .write()
            .map_err(|_| BPETokenizerError::InitializationFailed("Cache lock poisoned".to_string()))?;
        cache.clear();
        Ok(())
    }

    /// Get the current size of the BPE cache.
    pub fn cache_size(&self) -> Result<usize, MetalError> {
        let cache = self
            .bpe_cache
            .read()
            .map_err(|_| BPETokenizerError::InitializationFailed("Cache lock poisoned".to_string()))?;
        Ok(cache.len())
    }
}

#[path = "bpe.test.rs"]
mod tests;
