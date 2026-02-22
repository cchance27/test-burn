#![cfg(test)]

use super::Dtype;

#[test]
fn parse_fuzzy_handles_common_quant_aliases() {
    assert_eq!(Dtype::parse_fuzzy("Q4_0"), Some(Dtype::Q4_0));
    assert_eq!(Dtype::parse_fuzzy("q4_1"), Some(Dtype::Q4_1));
    assert_eq!(Dtype::parse_fuzzy("q6k"), Some(Dtype::Q6_K));
    assert_eq!(Dtype::parse_fuzzy("model-q8_0-gguf"), Some(Dtype::Q8_0));
    assert_eq!(Dtype::parse_fuzzy("dtype=bf16"), Some(Dtype::BF16));
}
