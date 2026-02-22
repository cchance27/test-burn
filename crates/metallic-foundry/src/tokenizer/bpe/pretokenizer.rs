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
pub(super) enum PreTokenizerKind {
    Gpt2Like,
    Llama3Like,
    Qwen2Like,
}

impl PreTokenizerKind {
    // GGUF metadata values are not standardized; keep aliases here.
    // Defaulting to GPT2-like maintains backward compatibility for older dumps
    // that omit `tokenizer.ggml.pre`.
    pub(super) fn from_metadata_name(name: Option<&str>) -> Self {
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

    pub(super) fn token_piece_pattern(self) -> &'static str {
        match self {
            Self::Gpt2Like => TOKEN_PIECE_RE_GPT2,
            Self::Llama3Like => TOKEN_PIECE_RE_LLAMA3,
            Self::Qwen2Like => TOKEN_PIECE_RE_QWEN2,
        }
    }

    pub(super) fn ignore_merges(self) -> bool {
        // Keep merge bypass plumbing available for future GGUF variants, but do not
        // enable it for current pre-tokenizers (including `llama-bpe`).
        // Only flip this if tokenizer-oracle parity demonstrates that merge ranks
        // must be bypassed for a specific `tokenizer.ggml.pre` family.
        false
    }
}

pub(super) fn pretokenizer_default_add_bos(name: Option<&str>) -> bool {
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
