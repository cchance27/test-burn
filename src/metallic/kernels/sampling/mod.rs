/// Maximum supported top-k for the GPU sampling kernel. Larger requests fall
/// back to the CPU implementation to avoid excessive per-thread stack usage.
pub const MAX_TOP_K: usize = 256;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
pub struct SamplingParams {
    pub vocab_size: u32,
    pub top_k: u32,
    pub top_p: f32,
    pub temperature: f32,
    pub random_u32: u32,
    pub _padding: u32,
}
