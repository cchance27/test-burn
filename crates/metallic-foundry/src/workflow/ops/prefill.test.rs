#![cfg(test)]

use super::apply_delta_cache_hit;

#[test]
fn partial_base_hit_uses_suffix_prefill() {
    let prompt_tokens_full = [10_u32, 11, 12, 13, 14, 15, 16, 17];
    let mut start_pos = 0usize;
    let mut prompt_tokens = &prompt_tokens_full[..];
    let mut token_source = "delta_input";
    let mut cache_hit_prefix_tokens = 0usize;
    let mut cache_lookup_path = "miss";

    apply_delta_cache_hit(
        &prompt_tokens_full,
        &mut start_pos,
        &mut prompt_tokens,
        &mut token_source,
        &mut cache_hit_prefix_tokens,
        &mut cache_lookup_path,
        5,
        "key_base",
        "delta_cache_key_base_suffix",
        "delta_cache_key_base_full_replay_last",
    );

    assert_eq!(start_pos, 5);
    assert_eq!(prompt_tokens, &prompt_tokens_full[5..]);
    assert_eq!(token_source, "delta_cache_key_base_suffix");
    assert_eq!(cache_hit_prefix_tokens, 5);
    assert_eq!(cache_lookup_path, "key_base");
}
