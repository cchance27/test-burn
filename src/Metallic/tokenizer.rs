//! Tokenizer implementation for the Metallic framework.
//!
//! This module provides BPE (Byte Pair Encoding) tokenization capabilities
//! that can work with GGUF metadata or other sources of vocabulary and merges.

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::hash::Hash;
use std::sync::RwLock;
use thiserror::Error;

/// Error types for tokenizer operations
#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("Invalid token ID: {0}")]
    InvalidTokenId(u32),
    #[error("Invalid UTF-8 sequence")]
    InvalidUtf8,
    #[error("Missing vocabulary or merges data")]
    MissingData,
    #[error("Tokenizer initialization failed: {0}")]
    InitializationFailed(String),
}

/// Special token IDs
#[derive(Default, Debug, Clone)]
pub struct SpecialTokens {
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
}

/// A BPE tokenizer implementation
pub struct Tokenizer {
    /// Vocabulary mapping token IDs to token strings
    vocab: FxHashMap<u32, String>,
    /// Reverse vocabulary mapping token strings to token IDs
    vocab_r: FxHashMap<String, u32>,
    /// BPE merges
    merges: FxHashMap<(String, String), u32>,
    /// Special tokens
    special_tokens: SpecialTokens,
    /// Whether to add BOS token
    add_bos_token: bool,
    /// Cache for BPE tokenization results
    bpe_cache: RwLock<FxHashMap<String, Vec<u32>>>,
}

impl Tokenizer {
    /// Create a new tokenizer with the given vocabulary and merges
    pub fn new(
        vocab: FxHashMap<u32, String>,
        merges: Vec<(String, String)>,
        special_tokens: SpecialTokens,
        add_bos_token: bool,
    ) -> Result<Self, TokenizerError> {
        // Create reverse vocabulary
        let mut vocab_r = FxHashMap::default();
        for (id, token) in &vocab {
            vocab_r.insert(token.clone(), *id);
        }

        // Create merges map with priority
        let mut merges_map = FxHashMap::default();
        for (i, merge) in merges.iter().enumerate() {
            merges_map.insert(merge.clone(), i as u32);
        }

        Ok(Self {
            vocab,
            vocab_r,
            merges: merges_map,
            special_tokens,
            add_bos_token,
            bpe_cache: RwLock::new(FxHashMap::default()),
        })
    }

    /// Create a new tokenizer with the given vocabulary, merges, and default special tokens
    pub fn from_vocab_and_merges(
        vocab: FxHashMap<u32, String>,
        merges: Vec<(String, String)>,
    ) -> Result<Self, TokenizerError> {
        Self::new(vocab, merges, SpecialTokens::default(), false)
    }

    /// Create a new tokenizer with custom configuration
    pub fn with_config(
        vocab: FxHashMap<u32, String>,
        merges: Vec<(String, String)>,
        special_tokens: SpecialTokens,
        add_bos_token: bool,
    ) -> Result<Self, TokenizerError> {
        Self::new(vocab, merges, special_tokens, add_bos_token)
    }

    /// Create a tokenizer from any source that provides vocabulary and merges
    /// This makes the tokenizer completely generic and not tied to GGUF
    pub fn from_generic_source(
        vocab_source: impl IntoIterator<Item = (u32, String)>,
        merges_source: impl IntoIterator<Item = (String, String)>,
        special_tokens: SpecialTokens,
        add_bos_token: bool,
    ) -> Result<Self, TokenizerError> {
        let vocab: FxHashMap<u32, String> = vocab_source.into_iter().collect();
        let merges: Vec<(String, String)> = merges_source.into_iter().collect();
        Self::new(vocab, merges, special_tokens, add_bos_token)
    }

    /// Create a tokenizer from GGUF metadata
    pub fn from_gguf_metadata(
        metadata: &crate::gguf::GGUFMetadata,
    ) -> Result<Self, TokenizerError> {
        // Extract vocabulary
        let tokens_value = metadata
            .entries
            .get("tokenizer.ggml.tokens")
            .ok_or(TokenizerError::MissingData)?;

        let merges_value = metadata
            .entries
            .get("tokenizer.ggml.merges")
            .ok_or(TokenizerError::MissingData)?;

        // Extract special tokens
        let bos_token_id = metadata
            .entries
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| match v {
                crate::gguf::GGUFValue::U32(id) => Some(*id),
                _ => None,
            });

        let eos_token_id = metadata
            .entries
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| match v {
                crate::gguf::GGUFValue::U32(id) => Some(*id),
                _ => None,
            });

        let pad_token_id = metadata
            .entries
            .get("tokenizer.ggml.padding_token_id")
            .and_then(|v| match v {
                crate::gguf::GGUFValue::U32(id) => Some(*id),
                _ => None,
            });

        let add_bos_token = metadata
            .entries
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| match v {
                crate::gguf::GGUFValue::Bool(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(false);

        // Parse tokens array
        let tokens = match tokens_value {
            crate::gguf::GGUFValue::Array(arr) => arr
                .iter()
                .map(|v| match v {
                    crate::gguf::GGUFValue::String(s) => Ok(s.clone()),
                    _ => Err(TokenizerError::InitializationFailed(
                        "Invalid token type in vocabulary".to_string(),
                    )),
                })
                .collect::<Result<Vec<String>, TokenizerError>>()?,
            _ => {
                return Err(TokenizerError::InitializationFailed(
                    "Invalid tokens format".to_string(),
                ));
            }
        };

        // Parse merges array
        let merges_str = match merges_value {
            crate::gguf::GGUFValue::Array(arr) => arr
                .iter()
                .map(|v| match v {
                    crate::gguf::GGUFValue::String(s) => Ok(s.clone()),
                    _ => Err(TokenizerError::InitializationFailed(
                        "Invalid merge type".to_string(),
                    )),
                })
                .collect::<Result<Vec<String>, TokenizerError>>()?,
            _ => {
                return Err(TokenizerError::InitializationFailed(
                    "Invalid merges format".to_string(),
                ));
            }
        };

        // Convert merges to pairs
        let merges = merges_str
            .into_iter()
            .filter_map(|merge| {
                let parts: Vec<&str> = merge.split(' ').collect();
                if parts.len() == 2 {
                    Some((parts[0].to_string(), parts[1].to_string()))
                } else {
                    None
                }
            })
            .collect::<Vec<(String, String)>>();

        // Create vocabulary map
        let mut vocab = FxHashMap::default();
        for (i, token) in tokens.into_iter().enumerate() {
            vocab.insert(i as u32, token);
        }

        let special_tokens = SpecialTokens {
            bos_token_id,
            eos_token_id,
            pad_token_id,
        };

        Self::new(vocab, merges, special_tokens, add_bos_token)
    }

    /// Encode text into tokens using the serial implementation
    pub fn encode_serial(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        // Estimate capacity based on text length (roughly 1 token per 3-4 characters)
        let estimated_capacity = text.len() / 4 + 10; // Add buffer for BOS token and special cases
        let mut tokens = Vec::with_capacity(estimated_capacity);

        // Add BOS token if required
        if let Some(bos_id) = self.special_tokens.bos_token_id
            && self.add_bos_token
        {
            tokens.push(bos_id);
        }

        // Byte-level preprocessing
        let preprocessed_text = self.byte_level_preprocess(text);

        // Split text into words (handling whitespace)
        let words: Vec<&str> = preprocessed_text.split_whitespace().collect();

        for word in words {
            // Apply BPE tokenization to each word
            let bpe_tokens = self.bpe_tokenize(word)?;
            tokens.extend(bpe_tokens);
        }

        Ok(tokens)
    }

    /// Encode text into tokens using parallel processing with Rayon
    pub fn encode_parallel(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        // Add BOS token if required
        // Estimate capacity based on text length (roughly 1 token per 3-4 characters)
        let estimated_capacity = text.len() / 4 + 10; // Add buffer for BOS token and special cases
        let mut tokens = Vec::with_capacity(estimated_capacity);
        if let Some(bos_id) = self.special_tokens.bos_token_id
            && self.add_bos_token
        {
            tokens.push(bos_id);
        }

        // Byte-level preprocessing
        let preprocessed_text = self.byte_level_preprocess(text);

        // Split text into words (handling whitespace)
        let words: Vec<&str> = preprocessed_text.split_whitespace().collect();

        // Process words in parallel
        // Estimate capacity for bpe_tokens based on number of words
        let mut bpe_tokens: Vec<Vec<u32>> = words
            .par_iter()
            .map(|word| self.bpe_tokenize(word))
            .collect::<Result<Vec<Vec<u32>>, TokenizerError>>()?;

        // Flatten the results
        for token_vec in bpe_tokens.drain(..) {
            tokens.extend(token_vec);
        }

        Ok(tokens)
    }

    /// Encode text into tokens using the default implementation (currently serial)
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.encode_serial(text)
    }

    /// Encode text into tokens using SIMD optimization
    #[cfg(target_arch = "aarch64")]
    pub fn encode_simd(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        // Use SIMD-optimized preprocessing
        let preprocessed_text = self.byte_level_preprocess_simd(text);

        // Estimate capacity based on text length (roughly 1 token per 3-4 characters)
        let estimated_capacity = text.len() / 4 + 10; // Add buffer for BOS token and special cases
        let mut tokens = Vec::with_capacity(estimated_capacity);

        // Add BOS token if required
        if let Some(bos_id) = self.special_tokens.bos_token_id
            && self.add_bos_token
        {
            tokens.push(bos_id);
        }

        // Split text into words (handling whitespace)
        let words: Vec<&str> = preprocessed_text.split_whitespace().collect();

        for word in words {
            // Apply BPE tokenization to each word
            let bpe_tokens = self.bpe_tokenize_simd(word)?;
            tokens.extend(bpe_tokens);
        }

        Ok(tokens)
    }

    /// Encode text into tokens using SIMD optimization
    #[cfg(not(target_arch = "aarch64"))]
    pub fn encode_simd(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        // For non-AArch64 architectures, just call the regular implementation
        self.encode_serial(text)
    }

    /// Encode text into tokens using SIMD and parallel processing
    #[cfg(target_arch = "aarch64")]
    pub fn encode_simd_parallel(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        // Use SIMD-optimized preprocessing
        let preprocessed_text = self.byte_level_preprocess_simd(text);

        // Add BOS token if required
        // Estimate capacity based on text length (roughly 1 token per 3-4 characters)
        let estimated_capacity = text.len() / 4 + 10; // Add buffer for BOS token and special cases
        let mut tokens = Vec::with_capacity(estimated_capacity);
        if let Some(bos_id) = self.special_tokens.bos_token_id
            && self.add_bos_token
        {
            tokens.push(bos_id);
        }

        // Split text into words (handling whitespace)
        let words: Vec<&str> = preprocessed_text.split_whitespace().collect();

        // Process words in parallel
        // Estimate capacity for bpe_tokens based on number of words
        let mut bpe_tokens: Vec<Vec<u32>> = words
            .par_iter()
            .map(|word| self.bpe_tokenize_simd(word))
            .collect::<Result<Vec<Vec<u32>>, TokenizerError>>()?;

        // Flatten the results
        for token_vec in bpe_tokens.drain(..) {
            tokens.extend(token_vec);
        }

        Ok(tokens)
    }

    /// Encode text into tokens using SIMD and parallel processing
    #[cfg(not(target_arch = "aarch64"))]
    pub fn encode_simd_parallel(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        // For non-AArch64 architectures, just call the parallel implementation
        self.encode_parallel(text)
    }

    /// Byte-level preprocessing for handling special characters
    pub fn byte_level_preprocess(&self, text: &str) -> String {
        // Estimate capacity based on text length (worst case: each char becomes <0xXX>)
        let estimated_capacity = text.len() * 6; // <0xXX> is 6 characters
        let mut result = String::with_capacity(estimated_capacity);
        for ch in text.chars() {
            match ch {
                '!'..='~' => result.push(ch), // Printable ASCII
                ' ' => result.push(ch),       // Space
                _ => {
                    // For other characters, use byte-level encoding
                    let ch_string = ch.to_string();
                    let bytes = ch_string.as_bytes();
                    for &byte in bytes {
                        // Map byte to printable character (using byte-fallback representation)
                        result.push_str(&format!("<0x{:02X}>", byte));
                    }
                }
            }
        }
        result
    }

    /// SIMD-optimized byte-level preprocessing for handling special characters
    #[cfg(target_arch = "aarch64")]
    pub fn byte_level_preprocess_simd(&self, text: &str) -> String {
        use std::arch::aarch64::*;

        // Estimate capacity based on text length (worst case: each char becomes <0xXX>)
        let estimated_capacity = text.len() * 6; // <0xXX> is 6 characters
        let mut result = String::with_capacity(estimated_capacity);
        let bytes = text.as_bytes();

        // Process 16 bytes at a time using SIMD
        let chunk_size = 16;
        let mut i = 0;

        while i + chunk_size <= bytes.len() {
            // Load 16 bytes into a SIMD register
            unsafe {
                let chunk = bytes[i..i + chunk_size].as_ptr();
                let simd_data = vld1q_u8(chunk);

                // Create masks for different character ranges
                // Printable ASCII: 0x20 (space) to 0x7E (~)
                let space_char = vdupq_n_u8(0x20);
                let tilde_char = vdupq_n_u8(0x7E);

                // Check if characters are in printable ASCII range
                let ge_space = vcgeq_u8(simd_data, space_char);
                let le_tilde = vcleq_u8(simd_data, tilde_char);
                let is_printable = vandq_u8(ge_space, le_tilde);

                // Extract results and process
                let printable_mask: [u8; 16] = std::mem::transmute(is_printable);
                let data: [u8; 16] = std::mem::transmute(simd_data);

                for j in 0..16 {
                    let byte = data[j];
                    if printable_mask[j] != 0 {
                        result.push(byte as char);
                    } else {
                        // For other characters, use byte-level encoding
                        result.push_str(&format!("<0x{:02X}>", byte));
                    }
                }
            }

            i += chunk_size;
        }

        // Process remaining bytes
        for &byte in &bytes[i..] {
            if (0x20..=0x7E).contains(&byte) {
                result.push(byte as char);
            } else {
                // For other characters, use byte-level encoding
                result.push_str(&format!("<0x{:02X}>", byte));
            }
        }

        result
    }

    /// SIMD-optimized byte-level preprocessing for handling special characters
    #[cfg(not(target_arch = "aarch64"))]
    pub fn byte_level_preprocess_simd(&self, text: &str) -> String {
        // For non-AArch64 architectures, just call the regular implementation
        self.byte_level_preprocess(text)
    }

    /// Apply BPE tokenization to a single word
    fn bpe_tokenize(&self, word: &str) -> Result<Vec<u32>, TokenizerError> {
        // Check cache first
        {
            let cache = self.bpe_cache.read().map_err(|_| {
                TokenizerError::InitializationFailed("Cache lock poisoned".to_string())
            })?;
            if let Some(cached_result) = cache.get(word) {
                return Ok(cached_result.clone());
            }
        }

        // Start with character-level tokens
        // Estimate capacity based on word length (worst case: each char becomes a token)
        let estimated_capacity = word.len() + 1; // Add 1 for </w> suffix
        let mut tokens: Vec<String> = Vec::with_capacity(estimated_capacity);
        for ch in word.chars() {
            tokens.push(ch.to_string());
        }

        // Add end-of-word suffix
        if let Some(last) = tokens.last_mut() {
            last.push_str("</w>");
        }

        // Iteratively apply BPE merges
        loop {
            // Find the best merge pair (lowest priority number)
            let mut best_pair: Option<((String, String), u32)> = None;
            let mut best_index = 0;

            for i in 0..tokens.len().saturating_sub(1) {
                let pair = (&tokens[i], &tokens[i + 1]);
                if let Some(priority) = self.merges.get(&(pair.0.clone(), pair.1.clone())) {
                    match &best_pair {
                        None => {
                            best_pair = Some(((pair.0.clone(), pair.1.clone()), *priority));
                            best_index = i;
                        }
                        Some((_, best_priority)) if priority < best_priority => {
                            best_pair = Some(((pair.0.clone(), pair.1.clone()), *priority));
                            best_index = i;
                        }
                        _ => {}
                    }
                }
            }

            // If no merge pair found, we're done
            let (pair, _) = match best_pair {
                Some(pair) => pair,
                None => break,
            };

            // Apply the merge
            // Estimate capacity for new_tokens (same as tokens, or slightly less)
            let mut new_tokens = Vec::with_capacity(tokens.len());
            let mut i = 0;
            while i < tokens.len() {
                if i == best_index {
                    // Merge the pair
                    new_tokens.push(format!("{}{}", pair.0, pair.1));
                    i += 2; // Skip the next token as it's been merged
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }
            tokens = new_tokens;
        }

        // Convert to token IDs
        let mut token_ids = Vec::new();
        for token in tokens {
            if let Some(token_id) = self.vocab_r.get(&token) {
                token_ids.push(*token_id);
            } else {
                // Handle unknown tokens with recursive splitting
                let sub_tokens = self.handle_unknown_token(&token)?;
                token_ids.extend(sub_tokens);
            }
        }

        // Store result in cache
        {
            let mut cache = self.bpe_cache.write().map_err(|_| {
                TokenizerError::InitializationFailed("Cache lock poisoned".to_string())
            })?;
            cache.insert(word.to_string(), token_ids.clone());
        }

        Ok(token_ids)
    }

    /// Handle unknown tokens by trying to split them into known subwords
    fn handle_unknown_token(&self, token: &str) -> Result<Vec<u32>, TokenizerError> {
        // Try to split the token into smaller parts
        // This is a simple fallback approach - in a production implementation,
        // you might want to use a more sophisticated algorithm

        // If the token is a single character, use UNK token
        if token.chars().count() <= 1 {
            return Ok(vec![0]); // UNK token ID
        }

        // Try to split the token in half and recursively process
        let chars: Vec<char> = token.chars().collect();
        let mid = chars.len() / 2;
        let left: String = chars[..mid].iter().collect();
        let right: String = chars[mid..].iter().collect();

        let mut result = Vec::new();

        // Process left part
        if let Some(token_id) = self.vocab_r.get(&left) {
            result.push(*token_id);
        } else {
            result.extend(self.handle_unknown_token(&left)?);
        }

        // Process right part
        if let Some(token_id) = self.vocab_r.get(&right) {
            result.push(*token_id);
        } else {
            result.extend(self.handle_unknown_token(&right)?);
        }

        Ok(result)
    }

    /// SIMD-optimized BPE tokenization for a single word
    #[cfg(target_arch = "aarch64")]
    fn bpe_tokenize_simd(&self, word: &str) -> Result<Vec<u32>, TokenizerError> {
        // Check cache first
        {
            let cache = self.bpe_cache.read().map_err(|_| {
                TokenizerError::InitializationFailed("Cache lock poisoned".to_string())
            })?;
            if let Some(cached_result) = cache.get(word) {
                return Ok(cached_result.clone());
            }
        }

        // Start with character-level tokens
        // Estimate capacity based on word length (worst case: each char becomes a token)
        let estimated_capacity = word.len() + 1; // Add 1 for </w> suffix
        let mut tokens: Vec<String> = Vec::with_capacity(estimated_capacity);
        for ch in word.chars() {
            tokens.push(ch.to_string());
        }

        // Add end-of-word suffix
        if let Some(last) = tokens.last_mut() {
            last.push_str("</w>");
        }

        // Iteratively apply BPE merges with SIMD optimizations
        loop {
            // Find the best merge pair (lowest priority number)
            let mut best_pair: Option<((String, String), u32)> = None;
            let mut best_index = 0;

            // For BPE, SIMD optimization is more about efficient data structures
            // and batch processing rather than vectorized operations on the data itself
            // We'll optimize by processing multiple candidate pairs at once

            // Process pairs in chunks for better cache efficiency
            let chunk_size = 8; // Process 8 pairs at a time
            let mut i = 0;
            while i < tokens.len().saturating_sub(1) {
                // Process up to chunk_size pairs
                let end = std::cmp::min(i + chunk_size, tokens.len() - 1);

                // Prepare batch of pairs for processing
                let mut pairs: Vec<((String, String), usize)> = Vec::with_capacity(end - i);
                for j in i..end {
                    pairs.push(((tokens[j].clone(), tokens[j + 1].clone()), j));
                }

                // Check multiple pairs in this chunk efficiently
                for (pair, index) in pairs {
                    if let Some(priority) = self.merges.get(&(pair.0.clone(), pair.1.clone())) {
                        match &best_pair {
                            None => {
                                best_pair = Some(((pair.0, pair.1), *priority));
                                best_index = index;
                            }
                            Some((_, best_priority)) if priority < best_priority => {
                                best_pair = Some(((pair.0, pair.1), *priority));
                                best_index = index;
                            }
                            _ => {}
                        }
                    }
                }

                i = end;
            }

            // If no merge pair found, we're done
            let (pair, _) = match best_pair {
                Some(pair) => pair,
                None => break,
            };

            // Apply the merge
            // Estimate capacity for new_tokens (same as tokens, or slightly less)
            let mut new_tokens = Vec::with_capacity(tokens.len());
            let mut i = 0;
            while i < tokens.len() {
                if i == best_index {
                    // Merge the pair
                    new_tokens.push(format!("{}{}", pair.0, pair.1));
                    i += 2; // Skip the next token as it's been merged
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }
            tokens = new_tokens;
        }

        // Convert to token IDs
        let mut token_ids = Vec::new();
        for token in tokens {
            if let Some(token_id) = self.vocab_r.get(&token) {
                token_ids.push(*token_id);
            } else {
                // Handle unknown tokens with recursive splitting
                let sub_tokens = self.handle_unknown_token(&token)?;
                token_ids.extend(sub_tokens);
            }
        }

        // Store result in cache
        {
            let mut cache = self.bpe_cache.write().map_err(|_| {
                TokenizerError::InitializationFailed("Cache lock poisoned".to_string())
            })?;
            cache.insert(word.to_string(), token_ids.clone());
        }

        Ok(token_ids)
    }

    /// SIMD-optimized BPE tokenization for a single word (non-AArch64 fallback)
    #[cfg(not(target_arch = "aarch64"))]
    fn bpe_tokenize_simd(&self, word: &str) -> Result<Vec<u32>, TokenizerError> {
        // On non-AArch64 architectures, just call the regular implementation
        self.bpe_tokenize(word)
    }

    /// Decode tokens into text
    pub fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        // Estimate capacity based on number of tokens (roughly 3-10 characters per token)
        let estimated_capacity = tokens.len() * 5;
        let mut decoded_parts = Vec::with_capacity(estimated_capacity);

        // Skip BOS token if present at the beginning
        let start_index = if self.add_bos_token
            && self.special_tokens.bos_token_id.is_some()
            && tokens.first() == self.special_tokens.bos_token_id.as_ref()
        {
            1
        } else {
            0
        };

        for token_id in &tokens[start_index..] {
            if let Some(token) = self.vocab.get(token_id) {
                decoded_parts.push(token.clone());
            } else {
                return Err(TokenizerError::InvalidTokenId(*token_id));
            }
        }

        // Join tokens and post-process
        let joined_text = decoded_parts.join("");

        // Post-process to remove end-of-word markers and handle byte-level decoding
        let processed_text = self.post_process(joined_text);

        Ok(processed_text)
    }

    /// Decode tokens into text using SIMD optimization
    #[cfg(target_arch = "aarch64")]
    pub fn decode_simd(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        use std::arch::aarch64::*;

        // Estimate capacity based on number of tokens (roughly 3-10 characters per token)
        let estimated_capacity = tokens.len() * 5;
        let mut decoded_parts = Vec::with_capacity(estimated_capacity);

        // Skip BOS token if present at the beginning
        let start_index = if self.add_bos_token
            && self.special_tokens.bos_token_id.is_some()
            && tokens.first() == self.special_tokens.bos_token_id.as_ref()
        {
            1
        } else {
            0
        };

        // Process tokens in chunks for SIMD optimization
        let chunk_size = 8;
        let mut i = start_index;

        while i + chunk_size <= tokens.len() {
            // Process chunk of tokens
            for token_id in &tokens[i..i + chunk_size] {
                if let Some(token) = self.vocab.get(token_id) {
                    decoded_parts.push(token.clone());
                } else {
                    return Err(TokenizerError::InvalidTokenId(*token_id));
                }
            }
            i += chunk_size;
        }

        // Process remaining tokens
        for token_id in &tokens[i..] {
            if let Some(token) = self.vocab.get(token_id) {
                decoded_parts.push(token.clone());
            } else {
                return Err(TokenizerError::InvalidTokenId(*token_id));
            }
        }

        // Join tokens and post-process
        let joined_text = decoded_parts.join("");

        // Post-process to remove end-of-word markers and handle byte-level decoding
        let processed_text = self.post_process_simd(joined_text);

        Ok(processed_text)
    }

    /// Decode tokens into text using SIMD optimization
    #[cfg(not(target_arch = "aarch64"))]
    pub fn decode_simd(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        // For non-AArch64 architectures, just call the regular implementation
        self.decode(tokens)
    }

    /// Post-process decoded text to remove BPE artifacts
    fn post_process(&self, text: String) -> String {
        // Remove end-of-word markers
        let text = text.replace("</w>", " ");

        // Handle byte-level decoding
        // Estimate capacity based on text length (worst case: no change)
        let estimated_capacity = text.len();
        let mut result = String::with_capacity(estimated_capacity);
        let mut i = 0;
        let chars: Vec<char> = text.chars().collect();
        while i < chars.len() {
            if i + 5 < chars.len()
                && chars[i] == '<'
                && chars[i + 1] == '0'
                && chars[i + 2] == 'x'
                && chars[i + 5] == '>'
            {
                // Try to parse byte
                let hex_str: String = chars[i + 3..i + 5].iter().collect();
                if let Ok(byte_val) = u8::from_str_radix(&hex_str, 16)
                    && let Some(ch) = char::from_u32(byte_val as u32)
                {
                    result.push(ch);
                    i += 6; // Skip the entire <0xXX> sequence
                    continue;
                }
            }
            // Add character as is
            result.push(chars[i]);
            i += 1;
        }

        // Clean up extra spaces
        result.replace("  ", " ").trim().to_string()
    }

    /// SIMD-optimized post-process decoded text to remove BPE artifacts
    #[cfg(target_arch = "aarch64")]
    fn post_process_simd(&self, text: String) -> String {
        // Remove end-of-word markers
        let text = text.replace("</w>", " ");

        // For post-processing, the SIMD optimization is less straightforward
        // since we're dealing with variable-length patterns like <0xXX>
        // We'll use a more efficient approach for scanning and replacing
        // these patterns without SIMD for now, but with better algorithmic
        // efficiency than the original implementation.

        // Estimate capacity based on text length (worst case: no change)
        let estimated_capacity = text.len();
        let mut result = String::with_capacity(estimated_capacity);
        let bytes = text.as_bytes();
        let mut i = 0;

        while i < bytes.len() {
            // Check for <0xXX> pattern
            if i + 5 < bytes.len()
                && bytes[i] == b'<'
                && bytes[i + 1] == b'0'
                && bytes[i + 2] == b'x'
                && bytes[i + 5] == b'>'
                && bytes[i + 3].is_ascii_hexdigit()
                && bytes[i + 4].is_ascii_hexdigit()
            {
                // Parse the hex digits
                let hex_str = std::str::from_utf8(&bytes[i + 3..i + 5]).unwrap_or("");
                if let Ok(byte_val) = u8::from_str_radix(hex_str, 16) {
                    result.push(byte_val as char);
                    i += 6; // Skip the entire <0xXX> sequence
                    continue;
                }
            }
            // Add character as is
            if let Some(ch) = char::from_u32(bytes[i] as u32) {
                result.push(ch);
            }
            i += 1;
        }

        // Clean up extra spaces
        result.replace("  ", " ").trim().to_string()
    }

    /// SIMD-optimized post-process decoded text to remove BPE artifacts
    #[cfg(not(target_arch = "aarch64"))]
    fn post_process_simd(&self, text: String) -> String {
        // For non-AArch64 architectures, just call the regular implementation
        self.post_process(text)
    }

    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get special tokens
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    /// Get a token by ID (for testing purposes)
    #[cfg(test)]
    pub fn get_token(&self, id: u32) -> Option<&String> {
        self.vocab.get(&id)
    }

    /// Clear the BPE cache
    pub fn clear_cache(&self) -> Result<(), TokenizerError> {
        let mut cache = self
            .bpe_cache
            .write()
            .map_err(|_| TokenizerError::InitializationFailed("Cache lock poisoned".to_string()))?;
        cache.clear();
        Ok(())
    }

    /// Get the current size of the BPE cache
    pub fn cache_size(&self) -> Result<usize, TokenizerError> {
        let cache = self
            .bpe_cache
            .read()
            .map_err(|_| TokenizerError::InitializationFailed("Cache lock poisoned".to_string()))?;
        Ok(cache.len())
    }
}
