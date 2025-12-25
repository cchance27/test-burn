//! Unified Softmax Kernel.
//!
//! Conditional kernel that selects between `SoftmaxVec` and `SoftmaxBlock`
//! based on sequence length.

use metallic_macros::ConditionalKernel;

use super::{SoftmaxBlock, SoftmaxVec};
use crate::types::TensorArg;

/// Unified Softmax kernel with automatic variant selection.
///
/// Dispatches to `SoftmaxVec` or `SoftmaxBlock` based on `seq_k`:
/// - Vec: 0-767, 896-1023, 1280-4095
/// - Block: 768-895, 1024-1279, 4096+
#[derive(ConditionalKernel, Clone)]
#[conditional(selector = "seq_k: usize")]
pub enum Softmax {
    /// Short sequences (0-767)
    #[when(seq_k.in_(0..=767))]
    VecShort(SoftmaxVec),

    /// Block for 768-895
    #[when(seq_k.in_(768..=895))]
    BlockMid1(SoftmaxBlock),

    /// Vec for 896-1023
    #[when(seq_k.in_(896..=1023))]
    VecMid(SoftmaxVec),

    /// Block for 1024-1279
    #[when(seq_k.in_(1024..=1279))]
    BlockMid2(SoftmaxBlock),

    /// Vec for 1280-4095
    #[when(seq_k.in_(1280..=4095))]
    VecLong(SoftmaxVec),

    /// Block for 4096+
    #[when(seq_k >= 4096)]
    BlockVeryLong(SoftmaxBlock),
}

impl Softmax {
    /// Create a new Softmax kernel, automatically selecting the best variant.
    pub fn new(input: &TensorArg, output: &TensorArg, rows_total: u32, seq_q: u32, seq_k: u32, causal: bool, query_offset: u32) -> Self {
        let batch = rows_total / seq_q; // Assuming divisible
        let seq_k_usize = seq_k as usize;

        // Use the generated select() to get the variant discriminant
        match Self::select(seq_k_usize) {
            SoftmaxVariant::VecShort => Softmax::VecShort(SoftmaxVec::new(input, output, rows_total, seq_q, seq_k, causal, query_offset)),
            SoftmaxVariant::VecMid => Softmax::VecMid(SoftmaxVec::new(input, output, rows_total, seq_q, seq_k, causal, query_offset)),
            SoftmaxVariant::VecLong => Softmax::VecLong(SoftmaxVec::new(input, output, rows_total, seq_q, seq_k, causal, query_offset)),
            SoftmaxVariant::BlockMid1 => Softmax::BlockMid1(SoftmaxBlock::new(input, output, batch, seq_q, seq_k, causal, query_offset)),
            SoftmaxVariant::BlockMid2 => Softmax::BlockMid2(SoftmaxBlock::new(input, output, batch, seq_q, seq_k, causal, query_offset)),
            SoftmaxVariant::BlockVeryLong => {
                Softmax::BlockVeryLong(SoftmaxBlock::new(input, output, batch, seq_q, seq_k, causal, query_offset))
            }
        }
    }
}
