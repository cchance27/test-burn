//! GemvV2 - Simplified GEMV using Stage composition.
//!
//! Uses the same pattern as SoftmaxV2:
//! - Simple reusable stages with `#[derive(Stage)]`
//! - CompoundKernel composition
//! - Parity testing against legacy

pub mod stages;
pub mod step;
