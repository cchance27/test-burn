mod gemv_n16_f16;
mod gemv_n1_f16;
mod gemv_n2_f16;
mod gemv_n4_f16;
mod gemv_n8_f16;

#[cfg(test)]
mod gemv_n8_f16_test;

pub use gemv_n1_f16::MatmulGemvSmallN1Op;
pub use gemv_n2_f16::MatmulGemvSmallN2Op;
pub use gemv_n4_f16::MatmulGemvSmallN4Op;
pub use gemv_n8_f16::MatmulGemvSmallN8Op;
pub use gemv_n16_f16::MatmulGemvSmallN16Op;
