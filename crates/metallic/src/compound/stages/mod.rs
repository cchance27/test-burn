//! Stage implementations for compound kernels.

mod epilogue;
mod gemv;
mod generic;
mod policy;

pub use epilogue::*;
pub use gemv::*;
pub use generic::*;
pub use policy::*;
