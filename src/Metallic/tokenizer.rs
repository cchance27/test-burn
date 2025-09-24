//! Tokenizer implementation for the Metallic framework.
//!
//! This module provides BPE (Byte Pair Encoding) tokenization capabilities
//! that can work with GGUF metadata or other sources of vocabulary and merges.

use rayon::prelude::*;
use rustc_hash::FxHashMap;
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
    /// Token types
    token_types: FxHashMap<u32, i32>,
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
        token_types: FxHashMap<u32, i32>,
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
            token_types,
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
        Self::new(vocab, merges, FxHashMap::default(), SpecialTokens::default(), false)
    }

    /// Create a new tokenizer with custom configuration
    pub fn with_config(
        vocab: FxHashMap<u32, String>,
        merges: Vec<(String, String)>,
        special_tokens: SpecialTokens,
        add_bos_token: bool,
    ) -> Result<Self, TokenizerError> {
        Self::new(vocab, merges, FxHashMap::default(), special_tokens, add_bos_token)
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
        Self::new(vocab, merges, FxHashMap::default(), special_tokens, add_bos_token)
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

        let token_types_value = metadata
            .entries
            .get("tokenizer.ggml.token_type");

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

        Self::new(vocab, merges, token_types_map, special_tokens, add_bos_token)
    }

    /// Encode text into tokens using the serial implementation
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.encode_serial(text)
    }

    /// Encode text into tokens using the serial implementation
    pub fn encode_serial(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let mut token_ids = Vec::new();
        if let Some(bos_id) = self.special_tokens.bos_token_id {
            if self.add_bos_token {
                token_ids.push(bos_id);
            }
        }

        let mut special_tokens = Vec::new();
        for token in self.vocab.values() {
            if token.starts_with("<|") && token.ends_with("|>") {
                special_tokens.push(token.clone());
            }
        }
        // Sort by length descending to match longest tokens first
        special_tokens.sort_by_key(|b| std::cmp::Reverse(b.len()));

        let mut parts = Vec::new();
        let mut last = 0;
        while last < text.len() {
            let remaining_text = &text[last..];
            // Find the earliest occurrence of any special token in remaining_text
            let mut found_pos: Option<(usize, &String)> = None;
            for token in &special_tokens {
                if let Some(pos) = remaining_text.find(token) {
                    match found_pos {
                        Some((prev_pos, _)) if pos >= prev_pos => {}
                        _ => found_pos = Some((pos, token)),
                    }
                }
            }
            if let Some((pos, token)) = found_pos {
                if pos > 0 {
                    parts.push(&text[last..last + pos]);
                }
                parts.push(&text[last + pos..last + pos + token.len()]);
                last += pos + token.len();
            } else {
                parts.push(remaining_text);
                break;
            }
        }

        for part in parts {
            if self.vocab_r.contains_key(part) && part.starts_with("<|") {
                token_ids.push(*self.vocab_r.get(part).unwrap());
                continue;
            }
            
            // BPE encode part
            let preprocessed: String = part.chars().map(|c| if c.is_whitespace() { 'Ġ' } else { c }).collect();
            let mut pieces: Vec<String> = preprocessed.chars().map(|c| c.to_string()).collect();
            if pieces.is_empty() {
                continue;
            }
            loop {
                let mut min_rank = u32::MAX;
                let mut merge_pos = None;

                for i in 0..pieces.len().saturating_sub(1) {
                    let pair_key = (pieces[i].clone(), pieces[i + 1].clone());
                    if let Some(&rank) = self.merges.get(&pair_key) {
                        if rank < min_rank {
                            min_rank = rank;
                            merge_pos = Some(i);
                        }
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

        Ok(token_ids)
    }

    /// Decode tokens into text
    pub fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        let mut bytes = Vec::new();
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
                let token_type = self.token_types.get(token_id).cloned().unwrap_or(1); // Default to normal
                if token_type == 6 { // Byte token
                    if token.starts_with("<0x") && token.ends_with('>') && token.len() == 6 {
                        if let Ok(byte) = u8::from_str_radix(&token[3..5], 16) {
                            bytes.push(byte);
                            println!("Token ID: {token_id}: Type: {token_type} TokenFromByte: {byte:?} {}", String::from_utf8_lossy(&[byte]));

                        } else {
                            unimplemented!("handle failed str_Radix")
                        }
                    } else {
                        unimplemented!("handle token missing <0x>")
                    }
                } else {
                    bytes.extend_from_slice(token.as_bytes());
                    println!("Token ID: {token_id}: Type: {token_type} TokenStr: {token}");                    
                }
            } else {
                return Err(TokenizerError::InvalidTokenId(*token_id));
            }
        }

        let decoded_text = String::from_utf8_lossy(&bytes).to_string();
        let processed_text = self.post_process(decoded_text);
        
        Ok(processed_text)
    }

    /// Post-process decoded text to remove BPE artifacts
    fn post_process(&self, text: String) -> String {
        // For GPT2, replace Ġ with space
        text.replace("Ġ",  " ")
            .replace("  ", " ")
            .trim()
            .to_string()
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

    /// Get a token by ID (for debugging purposes)
    pub fn get_token_debug(&self, id: u32) -> Option<&String> {
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