// Tokenizer implementation for the Metallic framework.
//!
//! This module provides BPE (Byte Pair Encoding) tokenization capabilities
//! that can work with GGUF metadata or other sources of vocabulary and merges.

use fancy_regex::Regex;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::{Arc, RwLock};
use thiserror::Error;
use unicode_normalization::UnicodeNormalization;

use crate::{gguf::file::GGUFMetadata, metallic::MetalError};

fn bytes_to_unicode() -> FxHashMap<u8, char> {
    let mut bs = (b'!'..=b'~').chain(b'\xa1'..=b'\xac').chain(b'\xae'..=b'\xff').collect::<Vec<_>>();
    let mut cs = bs.iter().map(|b| *b as u32).collect::<Vec<_>>();
    let mut n = 0;
    for b in 0..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.into_iter()
        .zip(cs.into_iter().map(|c| std::char::from_u32(c).unwrap()))
        .collect()
}

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
    #[error("Regex Tokenizer Errors: {0}")]
    RegexError(#[from] fancy_regex::Error),
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
    vocab: FxHashMap<u32, Arc<str>>,
    /// Reverse vocabulary mapping token strings to token IDs
    vocab_r: FxHashMap<String, u32>,
    /// BPE merges
    merges: FxHashMap<(String, String), u32>,
    /// Token types
    token_types: FxHashMap<u32, i32>,
    /// Special tokens
    special_tokens: SpecialTokens,
    /// Whether to add BOS token
    add_bos_token: bool,
    /// Cache for BPE tokenization results
    bpe_cache: RwLock<FxHashMap<String, Vec<u32>>>,
    /// ID-based BPE merges for optimized tokenization
    merges_ranks: FxHashMap<(u32, u32), u32>,
    /// ID-based BPE merge results for optimized tokenization
    merges_results: FxHashMap<(u32, u32), u32>,
    /// Array for O(1) byte-to-unicode mapping
    byte_encoder_array: [char; 256],
    /// Cache for single-character token lookups
    char_vocab: FxHashMap<char, u32>,
}

impl Tokenizer {
    /// Create a new tokenizer with the given vocabulary and merges
    pub fn new(
        vocab: FxHashMap<u32, String>,
        merges: Vec<(String, String)>,
        token_types: FxHashMap<u32, i32>,
        special_tokens: SpecialTokens,
        add_bos_token: bool,
    ) -> Result<Self, MetalError> {
        // Convert vocabulary into shared strings and build reverse map
        let mut vocab_arc = FxHashMap::default();
        let mut vocab_r = FxHashMap::default();
        for (id, token) in vocab.into_iter() {
            vocab_r.insert(token.clone(), id);
            vocab_arc.insert(id, Arc::<str>::from(token));
        }

        // Create merges map with priority
        let mut merges_map = FxHashMap::default();
        for (i, merge) in merges.iter().enumerate() {
            merges_map.insert(merge.clone(), i as u32);
        }

        // Create ID-based merge maps for optimized BPE
        let mut merges_ranks = FxHashMap::default();
        let mut merges_results = FxHashMap::default();
        for (i, merge) in merges.iter().enumerate() {
            let p1 = &merge.0;
            let p2 = &merge.1;
            let merged = format!("{}{}", p1, p2);
            if let (Some(&id1), Some(&id2), Some(&id_merged)) = (vocab_r.get(p1), vocab_r.get(p2), vocab_r.get(&merged)) {
                merges_ranks.insert((id1, id2), i as u32);
                merges_results.insert((id1, id2), id_merged);
            }
        }

        let byte_encoder = bytes_to_unicode();
        let mut byte_encoder_array = ['\0'; 256];
        for (b, c) in &byte_encoder {
            byte_encoder_array[*b as usize] = *c;
        }

        // Create a cache for single-character tokens
        let mut char_vocab = FxHashMap::default();
        for (token, &id) in &vocab_r {
            let mut chars = token.chars();
            if let (Some(c), None) = (chars.next(), chars.next()) {
                char_vocab.insert(c, id);
            }
        }

        Ok(Self {
            vocab: vocab_arc,
            vocab_r,
            merges: merges_map,
            token_types,
            special_tokens,
            add_bos_token,
            bpe_cache: RwLock::new(FxHashMap::default()),
            merges_ranks,
            merges_results,
            byte_encoder_array,
            char_vocab,
        })
    }

    /// Create a new tokenizer with the given vocabulary, merges, and default special tokens
    pub fn from_vocab_and_merges(vocab: FxHashMap<u32, String>, merges: Vec<(String, String)>) -> Result<Self, MetalError> {
        Self::new(vocab, merges, FxHashMap::default(), SpecialTokens::default(), false)
    }

    /// Create a new tokenizer with custom configuration
    pub fn with_config(
        vocab: FxHashMap<u32, String>,
        merges: Vec<(String, String)>,
        special_tokens: SpecialTokens,
        add_bos_token: bool,
    ) -> Result<Self, MetalError> {
        Self::new(vocab, merges, FxHashMap::default(), special_tokens, add_bos_token)
    }

    /// Create a tokenizer from any source that provides vocabulary and merges
    /// This makes the tokenizer completely generic and not tied to GGUF
    pub fn from_generic_source(
        vocab_source: impl IntoIterator<Item = (u32, String)>,
        merges_source: impl IntoIterator<Item = (String, String)>,
        special_tokens: SpecialTokens,
        add_bos_token: bool,
    ) -> Result<Self, MetalError> {
        let vocab: FxHashMap<u32, String> = vocab_source.into_iter().collect();
        let merges: Vec<(String, String)> = merges_source.into_iter().collect();
        Self::new(vocab, merges, FxHashMap::default(), special_tokens, add_bos_token)
    }

    /// Create a tokenizer from GGUF metadata
    pub fn from_gguf_metadata(metadata: &GGUFMetadata) -> Result<Self, MetalError> {
        // Extract vocabulary
        let tokens_value = metadata.entries.get("tokenizer.ggml.tokens").ok_or(TokenizerError::MissingData)?;

        let merges_value = metadata.entries.get("tokenizer.ggml.merges").ok_or(TokenizerError::MissingData)?;

        let token_types_value = metadata.entries.get("tokenizer.ggml.token_type");

        // Extract special tokens
        let bos_token_id = metadata.entries.get("tokenizer.ggml.bos_token_id").and_then(|v| match v {
            crate::gguf::GGUFValue::U32(id) => Some(*id),
            _ => None,
        });

        let eos_token_id = metadata.entries.get("tokenizer.ggml.eos_token_id").and_then(|v| match v {
            crate::gguf::GGUFValue::U32(id) => Some(*id),
            _ => None,
        });

        let pad_token_id = metadata.entries.get("tokenizer.ggml.padding_token_id").and_then(|v| match v {
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
                    _ => Err(TokenizerError::InitializationFailed("Invalid token type in vocabulary".to_string())),
                })
                .collect::<Result<Vec<String>, TokenizerError>>()?,
            _ => {
                return Err(TokenizerError::InitializationFailed("Invalid tokens format".to_string()).into());
            }
        };

        // Parse token types array
        let token_types_map = if let Some(crate::gguf::GGUFValue::Array(arr)) = token_types_value {
            arr.iter()
                .enumerate()
                .filter_map(|(i, v)| match v {
                    crate::gguf::GGUFValue::I32(t) => Some((i as u32, *t)),
                    _ => None,
                })
                .collect::<FxHashMap<u32, i32>>()
        } else {
            FxHashMap::default()
        };

        // Parse merges array
        let merges_str = match merges_value {
            crate::gguf::GGUFValue::Array(arr) => arr
                .iter()
                .map(|v| match v {
                    crate::gguf::GGUFValue::String(s) => Ok(s.clone()),
                    _ => Err(TokenizerError::InitializationFailed("Invalid merge type".to_string())),
                })
                .collect::<Result<Vec<String>, TokenizerError>>()?,
            _ => {
                return Err(TokenizerError::InitializationFailed("Invalid merges format".to_string()).into());
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

        Self::new(vocab, merges, token_types_map, special_tokens, add_bos_token)
    }

    /// Encode text into tokens (defaults to encode_simd now that we've benchmarked it as fastest in all lengths)
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, MetalError> {
        self.encode_simd(text)
    }

    fn process_pieces(&self, text: &str, mut processor: impl FnMut(&str) -> Result<(), MetalError>) -> Result<(), MetalError> {
        let special_re = Regex::new(r"<\|[^>]*\|>")?;
        let re = Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")?;
        let mut last = 0;
        for mat in special_re.find_iter(text) {
            let mat = mat?;
            if mat.start() > last {
                let subtext = &text[last..mat.start()];
                for submat in re.find_iter(subtext) {
                    processor(submat?.as_str())?;
                }
            }
            processor(mat.as_str())?;
            last = mat.end();
        }
        if last < text.len() {
            let subtext = &text[last..];
            for submat in re.find_iter(subtext) {
                processor(submat?.as_str())?;
            }
        }
        Ok(())
    }

    /// Encode text into tokens using the serial implementation
    pub fn encode_serial(&self, text: &str) -> Result<Vec<u32>, MetalError> {
        let mut norm_text = String::with_capacity(text.len());
        text.nfc().for_each(|c| norm_text.push(c));

        let mut token_ids = Vec::with_capacity(norm_text.len() / 2);
        if let Some(bos_id) = self.special_tokens.bos_token_id
            && self.add_bos_token
        {
            token_ids.push(bos_id);
        }

        self.process_pieces(&norm_text, |piece| {
            self.bpe_encode(piece, &mut token_ids);
            Ok(())
        })?;

        Ok(token_ids)
    }

    /// Encode text into tokens using the SIMD-optimized (ID-based) implementation
    pub fn encode_simd(&self, text: &str) -> Result<Vec<u32>, MetalError> {
        let mut norm_text = String::with_capacity(text.len());
        text.nfc().for_each(|c| norm_text.push(c));

        let mut token_ids = Vec::with_capacity(norm_text.len() / 2);
        if let Some(bos_id) = self.special_tokens.bos_token_id
            && self.add_bos_token
        {
            token_ids.push(bos_id);
        }

        self.process_pieces(&norm_text, |piece| {
            self.bpe_encode_ids(piece, &mut token_ids);
            Ok(())
        })?;

        Ok(token_ids)
    }

    /// Encode text into tokens using the parallel implementation
    pub fn encode_parallel(&self, text: &str) -> Result<Vec<u32>, MetalError> {
        let mut norm_text = String::with_capacity(text.len());
        text.nfc().for_each(|c| norm_text.push(c));

        let mut token_ids = Vec::with_capacity(norm_text.len() / 2);
        if let Some(bos_id) = self.special_tokens.bos_token_id
            && self.add_bos_token
        {
            token_ids.push(bos_id);
        }

        // In the parallel version, we must collect the pieces first before processing in parallel.
        // The streaming approach is not directly applicable with rayon's `par_iter`.
        let mut pieces = Vec::new();
        self.process_pieces(&norm_text, |piece| {
            pieces.push(piece.to_string());
            Ok(())
        })?;

        let tokens_from_pieces: Vec<Vec<u32>> = pieces
            .par_iter()
            .map(|piece| {
                let mut local_token_ids = Vec::new();
                self.bpe_encode_ids(piece, &mut local_token_ids);
                local_token_ids
            })
            .collect();

        token_ids.extend(tokens_from_pieces.into_iter().flatten());

        Ok(token_ids)
    }

    /// Encode text into tokens using the parallel SIMD-optimized (ID-based) implementation
    pub fn encode_simd_parallel(&self, text: &str) -> Result<Vec<u32>, MetalError> {
        let mut norm_text = String::with_capacity(text.len());
        text.nfc().for_each(|c| norm_text.push(c));

        let mut token_ids = Vec::with_capacity(norm_text.len() / 2);
        if let Some(bos_id) = self.special_tokens.bos_token_id
            && self.add_bos_token
        {
            token_ids.push(bos_id);
        }

        let mut pieces = Vec::new();
        self.process_pieces(&norm_text, |piece| {
            pieces.push(piece.to_string());
            Ok(())
        })?;

        let tokens_from_pieces: Vec<Vec<u32>> = pieces
            .par_iter()
            .map(|piece| {
                let mut local_token_ids = Vec::new();
                self.bpe_encode_ids(piece, &mut local_token_ids);
                local_token_ids
            })
            .collect();

        token_ids.extend(tokens_from_pieces.into_iter().flatten());

        Ok(token_ids)
    }

    fn bpe_encode(&self, text: &str, token_ids: &mut Vec<u32>) {
        if let Some(id) = self.vocab_r.get(text) {
            token_ids.push(*id);
            return;
        }

        let mut token_unicode = String::with_capacity(text.len());
        for &b in text.as_bytes() {
            token_unicode.push(self.byte_encoder_array[b as usize]);
        }

        let mut pieces: Vec<String> = token_unicode.chars().map(|c| c.to_string()).collect();
        if pieces.is_empty() {
            return;
        }
        loop {
            let mut min_rank = u32::MAX;
            let mut merge_pos = None;

            for i in 0..pieces.len().saturating_sub(1) {
                let pair_key = (pieces[i].clone(), pieces[i + 1].clone());
                if let Some(&rank) = self.merges.get(&pair_key)
                    && rank < min_rank
                {
                    min_rank = rank;
                    merge_pos = Some(i);
                }
            }

            if let Some(pos) = merge_pos {
                let merged = format!("{}{}", pieces[pos], pieces[pos + 1]);
                pieces.splice(pos..pos + 2, std::iter::once(merged));
            } else {
                break;
            }
        }
        for piece in pieces {
            if let Some(&id) = self.vocab_r.get(&piece) {
                token_ids.push(id);
            } else {
                token_ids.push(0); // UNK
            }
        }
    }

    fn bpe_encode_ids(&self, text: &str, token_ids: &mut Vec<u32>) {
        if let Some(id) = self.vocab_r.get(text) {
            token_ids.push(*id);
            return;
        }

        let mut token_unicode = String::with_capacity(text.len());
        for &b in text.as_bytes() {
            token_unicode.push(self.byte_encoder_array[b as usize]);
        }

        let mut piece_ids: Vec<u32> = Vec::with_capacity(token_unicode.len());
        piece_ids.extend(
            token_unicode.chars().map(|c| *self.char_vocab.get(&c).unwrap_or(&0)), // Use UNK for unknown chars
        );

        if piece_ids.is_empty() {
            return;
        }

        if piece_ids.len() == 1 {
            token_ids.extend(piece_ids);
            return;
        }

        loop {
            let mut min_rank = u32::MAX;
            let mut merge_pos = None;

            for i in 0..piece_ids.len().saturating_sub(1) {
                let pair_key = (piece_ids[i], piece_ids[i + 1]);
                if let Some(&rank) = self.merges_ranks.get(&pair_key)
                    && rank < min_rank
                {
                    min_rank = rank;
                    merge_pos = Some(i);
                }
            }

            if let Some(pos) = merge_pos {
                let pair_key = (piece_ids[pos], piece_ids[pos + 1]);
                let merged_id = self.merges_results[&pair_key];
                piece_ids.splice(pos..pos + 2, std::iter::once(merged_id));
            } else {
                break;
            }
        }
        token_ids.extend(piece_ids);
    }

    /// Decode tokens into text
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

        let token_type = self.token_types.get(&token_id).cloned().unwrap_or(1); // Default to normal
        if token_type == 3 {
            // Control token
            return Ok(None);
        }

        let token = self.vocab.get(&token_id).ok_or(TokenizerError::InvalidTokenId(token_id))?;

        let arc = match token_type {
            6 => {
                byte_scratch.clear();
                let token_str = token.as_ref();
                if token_str.starts_with("<0x") && token_str.ends_with('>') && token_str.len() == 6 {
                    let byte = u8::from_str_radix(&token_str[3..5], 16).map_err(|_| TokenizerError::InvalidTokenId(token_id))?;
                    byte_scratch.push(byte);
                    scratch.push_str(&String::from_utf8_lossy(byte_scratch));
                    Arc::<str>::from(std::mem::take(scratch))
                } else {
                    return Err(TokenizerError::InvalidTokenId(token_id).into());
                }
            }
            _ if Self::needs_post_process(token.as_ref()) => {
                scratch.push_str(token.as_ref());
                self.post_process_in_place(scratch);
                Arc::<str>::from(std::mem::take(scratch))
            }
            _ => Arc::clone(token),
        };

        Ok(Some(arc))
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
    pub fn get_token(&self, id: u32) -> Option<Arc<str>> {
        self.vocab.get(&id).cloned()
    }

    /// Get a token by ID (for debugging purposes)
    pub fn get_token_debug(&self, id: u32) -> Option<Arc<str>> {
        self.vocab.get(&id).cloned()
    }

    /// Clear the BPE cache
    pub fn clear_cache(&self) -> Result<(), MetalError> {
        let mut cache = self
            .bpe_cache
            .write()
            .map_err(|_| TokenizerError::InitializationFailed("Cache lock poisoned".to_string()))?;
        cache.clear();
        Ok(())
    }

    /// Get the current size of the BPE cache
    pub fn cache_size(&self) -> Result<usize, MetalError> {
        let cache = self
            .bpe_cache
            .read()
            .map_err(|_| TokenizerError::InitializationFailed("Cache lock poisoned".to_string()))?;
        Ok(cache.len())
    }
}
