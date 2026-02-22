use std::sync::OnceLock;

use metallic_env::{
    DEBUG_SAMPLE_LOGITS, DEBUG_SAMPLE_LOGITS_MAX_STEPS, DEBUG_SAMPLE_LOGITS_TOPN, DEBUG_STREAM_POLL, FoundryEnvVar, IGNORE_EOS_STOP, is_set
};

#[inline]
pub fn ignore_eos_stop_enabled() -> bool {
    static IGNORE: OnceLock<bool> = OnceLock::new();
    *IGNORE.get_or_init(|| IGNORE_EOS_STOP.get().ok().flatten().unwrap_or(false))
}

#[inline]
pub fn debug_stream_poll_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| DEBUG_STREAM_POLL.get().ok().flatten().unwrap_or(false))
}

#[inline]
pub fn debug_sample_logits_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| DEBUG_SAMPLE_LOGITS.get().ok().flatten().unwrap_or(false))
}

#[inline]
pub fn debug_sample_logits_top_n() -> usize {
    static TOP_N: OnceLock<usize> = OnceLock::new();
    *TOP_N.get_or_init(|| DEBUG_SAMPLE_LOGITS_TOPN.get().ok().flatten().filter(|v| *v > 0).unwrap_or(8))
}

#[inline]
pub fn debug_sample_logits_max_steps() -> u32 {
    static MAX_STEPS: OnceLock<u32> = OnceLock::new();
    *MAX_STEPS.get_or_init(|| DEBUG_SAMPLE_LOGITS_MAX_STEPS.get().ok().flatten().filter(|v| *v > 0).unwrap_or(64))
}

#[inline]
pub fn debug_workflow_ops_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| is_set(FoundryEnvVar::DebugWorkflowOps))
}

#[inline]
pub fn debug_tokenize_or_template_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| is_set(FoundryEnvVar::DebugTokenize) || is_set(FoundryEnvVar::DebugChatTemplate))
}

pub fn emit_tokenize_debug_snapshot(input_name: &str, mode: &str, text: &str, tokens: &[u32], decoded_head: &str) {
    let max_chars = 800usize;
    let shown = text.chars().take(max_chars).collect::<String>();
    let suffix = if text.chars().count() > max_chars { "...(truncated)" } else { "" };
    let head_n = 64usize.min(tokens.len());
    let token_head = &tokens[..head_n];

    eprintln!(
        "[metallic][debug] TokenizeOp mode={mode} input='{}' chars={} tokens={} head_ids={:?}\n[metallic][debug] decoded_head:\n{}{}",
        input_name,
        text.chars().count(),
        tokens.len(),
        token_head,
        decoded_head,
        if tokens.len() > head_n {
            "\n[metallic][debug] (decoded_head truncated to first 64 tokens)"
        } else {
            ""
        }
    );
    eprintln!("[metallic][debug] input_text_head:\n{}{}", shown, suffix);
}
