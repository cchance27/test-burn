use std::sync::Arc;

use fancy_regex::Regex;
use metallic_loader::ModelMetadata;
use rustc_hash::FxHashMap;

use super::{BPETokenizer, BPETokenizerError, SpecialTokens};
use crate::MetalError;

impl BPETokenizer {
    /// Create a new tokenizer with the given vocabulary and merges.
    pub fn new(
        vocab: FxHashMap<u32, String>,
        merges: Vec<(String, String)>,
        token_types: FxHashMap<u32, i32>,
        special_tokens: SpecialTokens,
        add_bos_token: bool,
        chat_template: Option<String>,
    ) -> Result<Self, MetalError> {
        Self::new_with_pretokenizer(vocab, merges, token_types, special_tokens, add_bos_token, chat_template, None)
    }

    pub(super) fn new_with_pretokenizer(
        vocab: FxHashMap<u32, String>,
        merges: Vec<(String, String)>,
        token_types: FxHashMap<u32, i32>,
        special_tokens: SpecialTokens,
        add_bos_token: bool,
        chat_template: Option<String>,
        pre_tokenizer_name: Option<&str>,
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

        let byte_encoder = super::bytes::bytes_to_unicode();
        let mut byte_encoder_array = ['\0'; 256];
        for (b, c) in &byte_encoder {
            byte_encoder_array[*b as usize] = *c;
        }

        let mut byte_decoder_lut = [0u16; 512];
        for (b, c) in &byte_encoder {
            let idx = *c as usize;
            if idx >= byte_decoder_lut.len() {
                return Err(BPETokenizerError::InitializationFailed("byte decoder LUT overflow".to_string()).into());
            }
            byte_decoder_lut[idx] = (*b as u16) + 1;
        }

        // Create a cache for single-character tokens
        let mut char_vocab = FxHashMap::default();
        for (token, &id) in &vocab_r {
            let mut chars = token.chars();
            if let (Some(c), None) = (chars.next(), chars.next()) {
                char_vocab.insert(c, id);
            }
        }

        let mut special_literal_tokens: Vec<Arc<str>> = Vec::new();
        for token in vocab_arc.values() {
            if token.starts_with('<') && token.ends_with('>') && !token.chars().any(|ch| ch.is_whitespace()) {
                special_literal_tokens.push(Arc::clone(token));
            }
        }
        special_literal_tokens.sort_by_key(|t| std::cmp::Reverse(t.len()));

        let special_token_re = Regex::new(r"<\|[^>]*\|>").map_err(BPETokenizerError::RegexError)?;
        let pre_kind = super::pretokenizer::PreTokenizerKind::from_metadata_name(pre_tokenizer_name);
        let token_piece_re = Regex::new(pre_kind.token_piece_pattern()).map_err(BPETokenizerError::RegexError)?;

        let chat_template = chat_template
            .as_deref()
            .filter(|v| !v.trim().is_empty())
            .map(crate::template::ChatTemplate::new);

        Ok(Self {
            vocab: vocab_arc,
            vocab_r,
            merges: merges_map,
            token_types,
            special_tokens,
            add_bos_token,
            chat_template,
            bpe_cache: std::sync::RwLock::new(FxHashMap::default()),
            merges_ranks,
            merges_results,
            byte_encoder_array,
            byte_decoder_lut,
            char_vocab,
            special_token_re,
            special_literal_tokens,
            token_piece_re,
            pre_tokenizer_kind: pre_kind,
            ignore_merges: pre_kind.ignore_merges(),
        })
    }

    /// Create a new tokenizer with the given vocabulary, merges, and default special tokens.
    pub fn from_vocab_and_merges(vocab: FxHashMap<u32, String>, merges: Vec<(String, String)>) -> Result<Self, MetalError> {
        Self::new(vocab, merges, FxHashMap::default(), SpecialTokens::default(), false, None)
    }

    /// Create a new tokenizer with custom configuration.
    pub fn with_config(
        vocab: FxHashMap<u32, String>,
        merges: Vec<(String, String)>,
        special_tokens: SpecialTokens,
        add_bos_token: bool,
    ) -> Result<Self, MetalError> {
        Self::new(vocab, merges, FxHashMap::default(), special_tokens, add_bos_token, None)
    }

    /// Create a tokenizer from any source that provides vocabulary and merges.
    /// This makes the tokenizer completely generic and not tied to GGUF.
    pub fn from_generic_source(
        vocab_source: impl IntoIterator<Item = (u32, String)>,
        merges_source: impl IntoIterator<Item = (String, String)>,
        special_tokens: SpecialTokens,
        add_bos_token: bool,
    ) -> Result<Self, MetalError> {
        let vocab: FxHashMap<u32, String> = vocab_source.into_iter().collect();
        let merges: Vec<(String, String)> = merges_source.into_iter().collect();
        Self::new(vocab, merges, FxHashMap::default(), special_tokens, add_bos_token, None)
    }

    /// Create a tokenizer from model metadata.
    pub fn from_metadata(metadata: &dyn ModelMetadata) -> Result<Self, MetalError> {
        // Extract vocabulary
        let tokens = metadata.tokenizer_tokens().ok_or(BPETokenizerError::MissingData)?;

        let merges_str = metadata.tokenizer_merges().ok_or(BPETokenizerError::MissingData)?;

        // Fallback for token types if not provided
        let token_types_value = metadata.get_array("tokenizer.ggml.token_type");

        // DEBT: We should properly clear this so that it isn't so hard tied to GGML.
        // Extract special tokens (standardize these too if possible, but GGUF keys are common in our current use cases)
        let bos_token_id = metadata
            .get_u32("tokenizer.ggml.bos_token_id")
            .or_else(|| metadata.get_u32("standard.bos_token_id"));
        let eos_token_id = metadata
            .get_u32("tokenizer.ggml.eos_token_id")
            .or_else(|| metadata.get_u32("standard.eos_token_id"));
        let pad_token_id = metadata
            .get_u32("tokenizer.ggml.padding_token_id")
            .or_else(|| metadata.get_u32("standard.padding_token_id"));

        let chat_template = metadata
            .get_string("tokenizer.chat_template")
            .or_else(|| metadata.get_string("standard.chat_template"))
            .map(|s| s.to_string());
        // Primary selector for tokenizer piece-splitting behavior.
        // Keep `standard.pre` fallback for non-GGUF metadata adapters.
        let pre_tokenizer_name = metadata
            .get_string("tokenizer.ggml.pre")
            .or_else(|| metadata.get_string("standard.pre"));

        let add_bos_token = metadata
            .get("tokenizer.ggml.add_bos_token")
            .or_else(|| metadata.get("standard.add_bos_token"))
            .and_then(|v| match v {
                metallic_loader::MetadataValue::Bool(b) => Some(b),
                _ => None,
            })
            .unwrap_or_else(|| super::pretokenizer::pretokenizer_default_add_bos(pre_tokenizer_name.as_deref()));

        // Parse token types array
        let token_types_map = if let Some(arr) = token_types_value {
            arr.into_iter()
                .enumerate()
                .filter_map(|(i, v)| v.as_i64().map(|t| (i as u32, t as i32)))
                .collect::<FxHashMap<u32, i32>>()
        } else {
            FxHashMap::default()
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

        Self::new_with_pretokenizer(
            vocab,
            merges,
            token_types_map,
            special_tokens,
            add_bos_token,
            chat_template,
            pre_tokenizer_name.as_deref(),
        )
    }
}
