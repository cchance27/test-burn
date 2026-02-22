#![cfg(test)]

use super::{effective_penalty_window, filter_prompt_tokens_for_penalty};

#[test]
fn eos_token_is_excluded_from_prompt_penalty_state() {
    let prompt = vec![151644, 8948, 151645, 123, 151645, 42];
    let filtered = filter_prompt_tokens_for_penalty(&prompt, Some(151645));
    assert_eq!(filtered.as_ref(), &[151644, 8948, 123, 42]);
}

#[test]
fn prompt_tokens_are_untouched_without_eos() {
    let prompt = vec![1, 2, 3, 4];
    let filtered = filter_prompt_tokens_for_penalty(&prompt, Some(151645));
    assert_eq!(filtered.as_ref(), prompt.as_slice());
}

#[test]
fn uncapped_generation_boosts_small_repeat_window() {
    assert_eq!(effective_penalty_window(64, 0), 256);
    assert_eq!(effective_penalty_window(300, 0), 300);
}

#[test]
fn capped_generation_keeps_requested_window() {
    assert_eq!(effective_penalty_window(64, 256), 64);
}
