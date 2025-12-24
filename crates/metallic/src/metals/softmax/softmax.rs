//! Unified Softmax Kernel.
//!
//! Conditional kernel that selects between `SoftmaxVec` and `SoftmaxBlock`
//! based on sequence length.

use super::{SoftmaxBlock, SoftmaxVariant, SoftmaxVec, select_softmax_variant};
use crate::{kernel_enum, types::TensorArg};

kernel_enum!(
    /// Unified Softmax kernel.
    pub enum Softmax {
        Vec(SoftmaxVec),
        Block(SoftmaxBlock),
    }
);

impl Softmax {
    /// Create a new Softmax kernel, automatically selecting the best variant.
    pub fn new(input: &TensorArg, output: &TensorArg, rows_total: u32, seq_q: u32, seq_k: u32, causal: bool, query_offset: u32) -> Self {
        let batch = rows_total / seq_q; // Assuming divisible

        match select_softmax_variant(seq_k as usize) {
            SoftmaxVariant::Vec => {
                let k = SoftmaxVec::new(input, output, rows_total, seq_q, seq_k, causal, query_offset);
                Softmax::Vec(k)
            }
            SoftmaxVariant::Block => {
                let k = SoftmaxBlock::new(input, output, batch, seq_q, seq_k, causal, query_offset);
                Softmax::Block(k)
            }
        }
    }
}
