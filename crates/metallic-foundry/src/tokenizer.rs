// Tokenizer implementation for the Metallic framework.
//!
//! This module provides BPE (Byte Pair Encoding) tokenization capabilities

use std::sync::{Arc, RwLock};

use fancy_regex::Regex;
use metallic_loader::ModelMetadata;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use thiserror::Error;
use unicode_normalization::UnicodeNormalization;

use crate::{
    MetalError, template::{ChatTemplate, Message}
};

// NOTE FOR MODEL BRING-UP:
// `tokenizer.ggml.pre` selects which regex family to use for initial piece splitting.
// If a new model emits nonsense despite correct weights/spec, confirm token-id parity first.
// Most bring-ups only need:
// 1) adding metadata alias mapping in `from_metadata_name`, and
// 2) adding/changing the regex pattern below.
// If parity still fails, the variant likely also needs merge or normalization behavior changes.
const TOKEN_PIECE_RE_GPT2: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
const TOKEN_PIECE_RE_LLAMA3: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const TOKEN_PIECE_RE_QWEN2: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)]
enum PreTokenizerKind {
    Gpt2Like,
    Llama3Like,
    Qwen2Like,
}

impl PreTokenizerKind {
    // GGUF metadata values are not standardized; keep aliases here.
    // Defaulting to GPT2-like maintains backward compatibility for older dumps
    // that omit `tokenizer.ggml.pre`.
    fn from_metadata_name(name: Option<&str>) -> Self {
        let Some(name) = name.map(str::trim).filter(|s| !s.is_empty()) else {
            return Self::Gpt2Like;
        };

        if [
            "llama3",
            "llama-v3",
            "llama-bpe",
            "smaug-bpe",
            "falcon3",
            "falcon-h1",
            "pixtral",
            "midm-2.0",
            "lfm2",
        ]
        .iter()
        .any(|v| name.eq_ignore_ascii_case(v))
        {
            return Self::Llama3Like;
        }

        if ["qwen2", "deepseek-r1-qwen", "kormo"].iter().any(|v| name.eq_ignore_ascii_case(v)) {
            return Self::Qwen2Like;
        }

        Self::Gpt2Like
    }

    fn token_piece_pattern(self) -> &'static str {
        match self {
            Self::Gpt2Like => TOKEN_PIECE_RE_GPT2,
            Self::Llama3Like => TOKEN_PIECE_RE_LLAMA3,
            Self::Qwen2Like => TOKEN_PIECE_RE_QWEN2,
        }
    }

    fn ignore_merges(self) -> bool {
        // Keep merge bypass plumbing available for future GGUF variants, but do not
        // enable it for current pre-tokenizers (including `llama-bpe`).
        // Only flip this if tokenizer-oracle parity demonstrates that merge ranks
        // must be bypassed for a specific `tokenizer.ggml.pre` family.
        false
    }
}

fn pretokenizer_default_add_bos(name: Option<&str>) -> bool {
    let Some(name) = name.map(str::trim).filter(|s| !s.is_empty()) else {
        return false;
    };

    [
        "llama3",
        "llama-v3",
        "llama-bpe",
        "falcon3",
        "falcon-h1",
        "pixtral",
        "midm-2.0",
        "lfm2",
        "tekken",
    ]
    .iter()
    .any(|v| name.eq_ignore_ascii_case(v))
}

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

/// Special token IDs
#[derive(Default, Debug, Clone)]
pub struct SpecialTokens {
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
}

/// A BPE tokenizer implementation
pub struct BPETokenizer {
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
    /// Dynamic chat template for formatting multi-turn prompts.
    chat_template: Option<ChatTemplate>,
    /// Cache for BPE tokenization results
    bpe_cache: RwLock<FxHashMap<String, Vec<u32>>>,
    /// ID-based BPE merges for optimized tokenization
    merges_ranks: FxHashMap<(u32, u32), u32>,
    /// ID-based BPE merge results for optimized tokenization
    merges_results: FxHashMap<(u32, u32), u32>,
    /// Array for O(1) byte-to-unicode mapping
    byte_encoder_array: [char; 256],
    /// LUT for inverse byte decoding (unicode codepoint -> byte+1).
    ///
    /// The GPT-2 byte encoder maps all 256 bytes into the range U+0021..U+00FF and U+0100..U+01FF,
    /// so a small fixed LUT is enough for the hot path.
    byte_decoder_lut: [u16; 512],
    /// Cache for single-character token lookups
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
    fn system_prompt_from_env() -> Option<String> {
        std::env::var("METALLIC_SYSTEM_PROMPT")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    #[inline]
    fn debug_dump_chat_template_render(kind: &str, formatted: &str) {
        if std::env::var("METALLIC_DEBUG_CHAT_TEMPLATE").is_ok() {
            eprintln!(
                "[metallic][debug] chat_template_render kind={} chars={}\n{}",
                kind,
                formatted.chars().count(),
                formatted
            );
        }
    }

    fn format_im_marker_messages(messages: &[Message], add_generation_prompt: bool) -> String {
        let mut cap = 0usize;
        for m in messages {
            cap = cap.saturating_add(m.role.len()).saturating_add(m.content.len()).saturating_add(32);
        }
        let mut s = String::with_capacity(cap.saturating_add(32));
        for m in messages {
            s.push_str("<|im_start|>");
            s.push_str(m.role.as_str());
            s.push('\n');
            s.push_str(m.content.as_str());
            s.push_str("<|im_end|>\n");
        }
        if add_generation_prompt {
            s.push_str("<|im_start|>assistant\n");
        }
        s
    }

    /// Canonical chat-message rendering entrypoint used by both tokenizer helpers and workflow `format_chat`.
    pub fn render_chat_messages(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
        template_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Result<(String, &'static str), MetalError> {
        if let Some(template) = &self.chat_template {
            let bos_token = self
                .special_tokens
                .bos_token_id
                .and_then(|id| self.vocab.get(&id).map(|s| s.as_ref()));
            let eos_token = self
                .special_tokens
                .eos_token_id
                .and_then(|id| self.vocab.get(&id).map(|s| s.as_ref()));

            let formatted = template.render(messages, bos_token, eos_token, add_generation_prompt, template_kwargs)?;
            Self::debug_dump_chat_template_render("messages", &formatted);
            return Ok((formatted, "chat_template"));
        }

        let has_im_markers = self.has_token("<|im_start|>") && self.has_token("<|im_end|>");
        if has_im_markers {
            let formatted = Self::format_im_marker_messages(messages, add_generation_prompt);
            Self::debug_dump_chat_template_render("im_marker_fallback", &formatted);
            return Ok((formatted, "im_marker_fallback"));
        }

        Ok((
            messages.iter().map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n"),
            "plain_join",
        ))
    }

    /// Create a new tokenizer with the given vocabulary and merges
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

    fn new_with_pretokenizer(
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

        let byte_encoder = bytes_to_unicode();
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
        let pre_kind = PreTokenizerKind::from_metadata_name(pre_tokenizer_name);
        let token_piece_re = Regex::new(pre_kind.token_piece_pattern()).map_err(BPETokenizerError::RegexError)?;

        let chat_template = chat_template.as_deref().filter(|v| !v.trim().is_empty()).map(ChatTemplate::new);

        Ok(Self {
            vocab: vocab_arc,
            vocab_r,
            merges: merges_map,
            token_types,
            special_tokens,
            add_bos_token,
            chat_template,
            bpe_cache: RwLock::new(FxHashMap::default()),
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

    /// Create a new tokenizer with the given vocabulary, merges, and default special tokens
    pub fn from_vocab_and_merges(vocab: FxHashMap<u32, String>, merges: Vec<(String, String)>) -> Result<Self, MetalError> {
        Self::new(vocab, merges, FxHashMap::default(), SpecialTokens::default(), false, None)
    }

    /// Create a new tokenizer with custom configuration
    pub fn with_config(
        vocab: FxHashMap<u32, String>,
        merges: Vec<(String, String)>,
        special_tokens: SpecialTokens,
        add_bos_token: bool,
    ) -> Result<Self, MetalError> {
        Self::new(vocab, merges, FxHashMap::default(), special_tokens, add_bos_token, None)
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
        Self::new(vocab, merges, FxHashMap::default(), special_tokens, add_bos_token, None)
    }

    /// Create a tokenizer from model metadata
    pub fn from_metadata(metadata: &dyn ModelMetadata) -> Result<Self, MetalError> {
        // Extract vocabulary
        let tokens = metadata.tokenizer_tokens().ok_or(BPETokenizerError::MissingData)?;

        let merges_str = metadata.tokenizer_merges().ok_or(BPETokenizerError::MissingData)?;

        // Fallback for token types if not provided
        let token_types_value = metadata.get_array("tokenizer.ggml.token_type");

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
            .unwrap_or_else(|| pretokenizer_default_add_bos(pre_tokenizer_name.as_deref()));

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

    /// Encode text into tokens (defaults to encode_simd now that we've benchmarked it as fastest in all lengths)
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, MetalError> {
        self.encode_simd(text)
    }

    fn process_non_special_subtext(
        &self,
        subtext: &str,
        processor: &mut impl FnMut(&str) -> Result<(), MetalError>,
    ) -> Result<(), MetalError> {
        if matches!(self.pre_tokenizer_kind, PreTokenizerKind::Llama3Like) {
            return self.process_llama3_like_pieces(subtext, processor);
        }

        for submat in self.token_piece_re.find_iter(subtext) {
            processor(submat?.as_str())?;
        }
        Ok(())
    }

    fn process_llama3_like_pieces(
        &self,
        subtext: &str,
        processor: &mut impl FnMut(&str) -> Result<(), MetalError>,
    ) -> Result<(), MetalError> {
        if subtext.is_empty() {
            return Ok(());
        }

        let chars: Vec<char> = subtext.chars().collect();
        let mut char_starts: Vec<usize> = subtext.char_indices().map(|(idx, _)| idx).collect();
        char_starts.push(subtext.len());

        let mut emit_piece = |start_char: usize, end_char: usize| -> Result<(), MetalError> {
            if end_char <= start_char {
                return Ok(());
            }
            let start_byte = char_starts[start_char];
            let end_byte = char_starts[end_char];
            processor(&subtext[start_byte..end_byte])
        };

        let n = chars.len();
        let mut pos = 0usize;
        while pos < n {
            let start = pos;
            let c = chars[pos];

            if c == '\'' && pos + 1 < n {
                let c1 = chars[pos + 1].to_ascii_lowercase();
                if c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd' {
                    pos += 2;
                    emit_piece(start, pos)?;
                    continue;
                }
                if pos + 2 < n {
                    let c2 = chars[pos + 2].to_ascii_lowercase();
                    if ((c1 == 'v' || c1 == 'r') && c2 == 'e') || (c1 == 'l' && c2 == 'l') {
                        pos += 3;
                        emit_piece(start, pos)?;
                        continue;
                    }
                }
            }

            if c != '\r' && c != '\n' && !c.is_numeric() && (c.is_alphabetic() || (pos + 1 < n && chars[pos + 1].is_alphabetic())) {
                pos += 1;
                while pos < n && chars[pos].is_alphabetic() {
                    pos += 1;
                }
                emit_piece(start, pos)?;
                continue;
            }

            if c.is_numeric() {
                let mut group_start = pos;
                while pos < n && chars[pos].is_numeric() {
                    pos += 1;
                    if pos - group_start >= 3 {
                        emit_piece(group_start, pos)?;
                        group_start = pos;
                    }
                }
                emit_piece(group_start, pos)?;
                continue;
            }

            let mut test_pos = pos;
            if c == ' ' && pos + 1 < n {
                test_pos = pos + 1;
            }
            let punct_like =
                (test_pos < n) && !chars[test_pos].is_whitespace() && !chars[test_pos].is_alphabetic() && !chars[test_pos].is_numeric();
            if punct_like {
                if c == ' ' {
                    pos += 1;
                }
                while pos < n {
                    let ch = chars[pos];
                    if ch.is_whitespace() || ch.is_alphabetic() || ch.is_numeric() {
                        break;
                    }
                    pos += 1;
                }
                while pos < n && (chars[pos] == '\r' || chars[pos] == '\n') {
                    pos += 1;
                }
                emit_piece(start, pos)?;
                continue;
            }

            let mut num_whitespace = 0usize;
            let mut last_end_r_or_n = 0usize;
            while pos + num_whitespace < n && chars[pos + num_whitespace].is_whitespace() {
                let ch = chars[pos + num_whitespace];
                if ch == '\r' || ch == '\n' {
                    last_end_r_or_n = pos + num_whitespace + 1;
                }
                num_whitespace += 1;
            }

            if last_end_r_or_n > 0 {
                pos = last_end_r_or_n;
                emit_piece(start, pos)?;
                continue;
            }
            if num_whitespace > 1 && pos + num_whitespace < n {
                pos += num_whitespace - 1;
                emit_piece(start, pos)?;
                continue;
            }
            if num_whitespace > 0 {
                pos += num_whitespace;
                emit_piece(start, pos)?;
                continue;
            }

            pos += 1;
            emit_piece(start, pos)?;
        }

        Ok(())
    }

    fn process_pieces(&self, text: &str, mut processor: impl FnMut(&str) -> Result<(), MetalError>) -> Result<(), MetalError> {
        let mut start = 0usize;
        let mut pos = 0usize;

        while pos < text.len() {
            let rest = &text[pos..];
            let mut matched_special: Option<&str> = None;
            if rest.as_bytes().first().copied() == Some(b'<') {
                for tok in &self.special_literal_tokens {
                    let tok = tok.as_ref();
                    if rest.starts_with(tok) {
                        matched_special = Some(tok);
                        break;
                    }
                }
                if matched_special.is_none()
                    && let Some(mat) = self.special_token_re.find_iter(rest).next()
                {
                    let mat = mat?;
                    if mat.start() == 0 {
                        matched_special = Some(mat.as_str());
                    }
                }
            }

            if let Some(tok) = matched_special {
                if pos > start {
                    let subtext = &text[start..pos];
                    self.process_non_special_subtext(subtext, &mut processor)?;
                }
                processor(tok)?;
                pos += tok.len();
                start = pos;
                continue;
            }

            let ch_len = rest.chars().next().map(char::len_utf8).unwrap_or(1);
            pos += ch_len;
        }

        if start < text.len() {
            let subtext = &text[start..];
            self.process_non_special_subtext(subtext, &mut processor)?;
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

        if self.ignore_merges {
            if let Some(&id) = self.vocab_r.get(&token_unicode) {
                token_ids.push(id);
                return;
            }
            for ch in token_unicode.chars() {
                let mut utf8 = [0_u8; 4];
                let encoded = ch.encode_utf8(&mut utf8);
                if let Some(&id) = self.vocab_r.get(encoded) {
                    token_ids.push(id);
                } else {
                    token_ids.push(0);
                }
            }
            return;
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

        if self.ignore_merges {
            if let Some(&id) = self.vocab_r.get(&token_unicode) {
                token_ids.push(id);
                return;
            }
            token_ids.extend(token_unicode.chars().map(|c| *self.char_vocab.get(&c).unwrap_or(&0)));
            return;
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

    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get special tokens
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

    pub fn set_chat_template(&mut self, template: String) {
        self.chat_template = Some(ChatTemplate::new(&template));
    }

    /// Format a single-turn chat prompt using the dynamic template.
    ///
    /// If the tokenizer does not have a chat template, it returns the prompt as-is.
    pub fn format_single_turn_chat_prompt_with_kwargs(
        &self,
        prompt: &str,
        template_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Result<String, MetalError> {
        // Fast-path: if the prompt is already chat-formatted, do not wrap it again.
        if prompt.contains("<|im_start|>") {
            return Ok(prompt.to_string());
        }

        if self.chat_template.is_some() {
            let mut messages = Vec::with_capacity(2);
            if let Some(system) = Self::system_prompt_from_env() {
                messages.push(Message {
                    role: "system".to_string(),
                    content: system,
                });
            }
            messages.push(Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            });
            let (formatted, _path) = self.render_chat_messages(&messages, true, template_kwargs)?;
            return Ok(formatted);
        }

        // Fallback: if the tokenizer doesn't provide a template, pass raw prompt through.
        Ok(prompt.to_string())
    }

    pub fn format_single_turn_chat_prompt(&self, prompt: &str) -> Result<String, MetalError> {
        self.format_single_turn_chat_prompt_with_kwargs(prompt, None)
    }

    pub fn encode_single_turn_chat_prompt_with_kwargs(
        &self,
        prompt: &str,
        template_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Result<Vec<u32>, MetalError> {
        let formatted = self.format_single_turn_chat_prompt_with_kwargs(prompt, template_kwargs)?;
        self.encode(&formatted)
    }

    pub fn encode_single_turn_chat_prompt(&self, prompt: &str) -> Result<Vec<u32>, MetalError> {
        self.encode_single_turn_chat_prompt_with_kwargs(prompt, None)
    }

    /// Format a continuation chat prompt using the dynamic template.
    ///
    /// This adds the user prompt and opens the assistant turn.
    pub fn format_chat_continuation_prompt_with_kwargs(
        &self,
        prompt: &str,
        template_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Result<String, MetalError> {
        if self.chat_template.is_none() {
            return Ok(prompt.to_string());
        }

        let messages = vec![Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];

        let (formatted, _path) = self.render_chat_messages(&messages, true, template_kwargs)?;
        Ok(formatted)
    }

    pub fn format_chat_continuation_prompt(&self, prompt: &str) -> Result<String, MetalError> {
        self.format_chat_continuation_prompt_with_kwargs(prompt, None)
    }

    pub fn encode_chat_continuation_prompt_with_kwargs(
        &self,
        prompt: &str,
        template_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Result<Vec<u32>, MetalError> {
        let formatted = self.format_chat_continuation_prompt_with_kwargs(prompt, template_kwargs)?;
        self.encode(&formatted)
    }

    pub fn encode_chat_continuation_prompt(&self, prompt: &str) -> Result<Vec<u32>, MetalError> {
        self.encode_chat_continuation_prompt_with_kwargs(prompt, None)
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
            .map_err(|_| BPETokenizerError::InitializationFailed("Cache lock poisoned".to_string()))?;
        cache.clear();
        Ok(())
    }

    /// Get the current size of the BPE cache
    pub fn cache_size(&self) -> Result<usize, MetalError> {
        let cache = self
            .bpe_cache
            .read()
            .map_err(|_| BPETokenizerError::InitializationFailed("Cache lock poisoned".to_string()))?;
        Ok(cache.len())
    }
}

#[cfg(test)]
mod tests {
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
}
