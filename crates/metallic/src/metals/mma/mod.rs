//! MMA (Matrix Multiply-Accumulate) primitives for GEMM.
//!
//! This module provides reusable primitives for tiled matrix multiplication:
//! - `TileLoaderStage` - Policy-aware tile loading from device to threadgroup
//! - `MmaLoopStage` - Simdgroup MMA using Apple's AMX hardware
//!
//! These primitives are composable Lego blocks that can be combined
//! with layout stages, epilogue stages, etc. to build complete GEMM kernels.
//!
//! # Policy Abstraction
//!
//! The `TileLoaderStage` uses Policy templates for transparent F16/Q8/Q4
//! dequantization. The `MmaLoopStage` is policy-agnostic and works on
//! already-dequantized threadgroup memory.

pub mod stages;

/// Metal source for tile loader primitive.
pub const TILE_LOADER_METAL: &str = include_str!("tile_loader.metal");

/// Metal source for MMA primitive.
pub const MMA_METAL: &str = include_str!("mma.metal");
