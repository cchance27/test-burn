//! GEMM (General Matrix Multiply) compound kernel for Foundry.
//!
//! This module provides a fused GEMM kernel for prefill/prompt processing
//! where M > 1. It uses the MMA primitives for hardware-accelerated
//! matrix multiplication.
//!
//! # Architecture
//!
//! The GEMM kernel is composed from reusable primitives:
//! - `TileLayoutStage` - Tile indexing and configuration
//! - `TileLoadAStage` - Load A (activations) to threadgroup
//! - `TileLoadBStage` - Load B (weights) with Policy dequant
//! - `MmaLoopStage` - Simdgroup MMA computation
//! - `GemmEpilogueStage` - Alpha/beta/bias and output write
//!
//! # Policy Abstraction
//!
//! Matrix B (weights) can be F16 or quantized (Q8, Q4, etc.).
//! The Policy template handles dequantization transparently.

pub mod step;

pub use step::{CompiledGemmV2Step, GemmV2Step};
