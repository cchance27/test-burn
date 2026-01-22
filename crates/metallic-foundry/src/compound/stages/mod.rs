//! Stage implementations for compound kernels.

mod epilogue;
mod generic;
mod layout;
mod policy;
mod simd;

pub use epilogue::*;
pub use generic::*;
pub use layout::*;
pub use policy::*;
pub use simd::*;
