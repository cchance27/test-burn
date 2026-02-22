use std::sync::OnceLock;

use metallic_env::GEMV_FORCE_SIMD_SUM_REDUCE;

use crate::compound::{BufferArg, Stage};

/// Operation type for SIMD reduction.
#[derive(Debug, Clone, Copy)]
pub enum SimdOp {
    Max,
    Sum,
}

impl SimdOp {
    fn func_name(&self) -> &'static str {
        match self {
            SimdOp::Max => "block_reduce_max",
            SimdOp::Sum => "block_reduce_sum",
        }
    }
}

/// A stage that performs a block-wide SIMD reduction.
///
/// Wraps `simd.metal` primitives.
#[derive(Debug, Clone)]
pub struct SimdStage {
    op: SimdOp,
    input_var: String,
    output_var: String,
    threads: u32,
    dtype: String,
}

impl SimdStage {
    pub fn new(op: SimdOp, input_var: impl Into<String>, output_var: impl Into<String>) -> Self {
        Self {
            op,
            input_var: input_var.into(),
            output_var: output_var.into(),
            threads: 256, // Default for now, could be dynamic
            dtype: "float".to_string(),
        }
    }

    pub fn reduce_max(input_var: impl Into<String>, output_var: impl Into<String>) -> Self {
        Self::new(SimdOp::Max, input_var, output_var)
    }

    pub fn reduce_sum(input_var: impl Into<String>, output_var: impl Into<String>) -> Self {
        Self::new(SimdOp::Sum, input_var, output_var)
    }

    pub fn with_threads(mut self, threads: u32) -> Self {
        self.threads = threads;
        self
    }

    pub fn with_dtype(mut self, dtype: &str) -> Self {
        self.dtype = dtype.to_string();
        self
    }
}

impl Stage for SimdStage {
    fn includes(&self) -> Vec<&'static str> {
        // No external includes needed as we inline the code
        vec![]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        // Reductions use threadgroup memory, not device buffers
        vec![]
    }

    fn emit(&self, _prev_output: &str) -> (String, String) {
        let func = self.op.func_name();

        // Define threadgroup memory name based on op to allow multiple distinct reductions
        let shared_mem_name = format!("shared_{}", self.op.func_name().replace("block_reduce_", ""));

        // Declare shared memory at start of block
        let code = format!(
            "    threadgroup {dtype} {shared_mem}[{threads}];\n    // SIMD {op:?} Reduction\n    {dtype} {out} = {func}<{dtype}, {threads}>({in_var}, {shared_mem}, tid);",
            op = self.op,
            dtype = self.dtype,
            out = self.output_var,
            func = func,
            threads = self.threads,
            in_var = self.input_var,
            shared_mem = shared_mem_name
        );

        (self.output_var.clone(), code)
    }

    fn struct_defs(&self) -> String {
        // Inline the SIMD primitives directly to avoid include path issues at runtime
        // The path is relative to this file (crates/metallic/src/compound/stages/simd.rs)
        // Fixed: Renamed local vars to avoid shadowing function names
        r#"
#ifndef SIMD_METAL_H
#define SIMD_METAL_H

#include <metal_stdlib>
using namespace metal;

/// Optimized block-wide max reduction using SIMD intrinsics.
template<typename T, uint THREADS>
inline T block_reduce_max(T val, threadgroup T* shared_mem, uint tid) {
    // 1. SIMD-group reduction
    T s_max = simd_max(val);

    // 2. Write representative to shared memory
    if (tid % 32 == 0) {
        shared_mem[tid / 32] = s_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Thread 0 aggregates SIMD group results
    if (tid == 0) {
        T final_max = shared_mem[0];
        for (uint i = 1; i < THREADS / 32; ++i) {
            final_max = max(final_max, shared_mem[i]);
        }
        shared_mem[0] = final_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return shared_mem[0];
}

/// Optimized block-wide sum reduction using SIMD intrinsics.
template<typename T, uint THREADS>
inline T block_reduce_sum(T val, threadgroup T* shared_mem, uint tid) {
    // 1. SIMD-group reduction
    T s_sum = simd_sum(val);

    // 2. Write representative to shared memory
    if (tid % 32 == 0) {
        shared_mem[tid / 32] = s_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Thread 0 aggregates SIMD group results
    if (tid == 0) {
        T final_sum = shared_mem[0];
        for (uint i = 1; i < THREADS / 32; ++i) {
            final_sum += shared_mem[i];
        }
        shared_mem[0] = final_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return shared_mem[0];
}

#endif // SIMD_METAL_H
"#
        .to_string()
    }
}

// =============================================================================
// WarpReduceStage - SIMD-only warp reduction (no shared memory)
// =============================================================================

/// Reduction operation type.
#[derive(Debug, Clone, Copy)]
pub enum WarpReduceOp {
    Sum,
    Max,
}

/// A stage that performs warp-level SIMD reduction without shared memory.
///
/// Unlike `SimdStage` which uses threadgroup memory for cross-warp reduction,
/// this uses only SIMD shuffle intrinsics (`simd_shuffle_xor`).
///
/// Use when each warp processes a separate output (e.g., warp-per-row GEMV).
#[derive(Debug, Clone)]
pub struct WarpReduceStage {
    op: WarpReduceOp,
    input_var: String,
    output_var: String,
    /// SIMD width (32 for Apple Silicon, 64 for some AMD)
    simd_width: u32,
    dtype: String,
}

#[inline]
fn force_simd_sum_reduce() -> bool {
    static FORCE: OnceLock<bool> = OnceLock::new();
    *FORCE.get_or_init(|| GEMV_FORCE_SIMD_SUM_REDUCE.get().ok().flatten().unwrap_or(false))
}

impl WarpReduceStage {
    pub fn new(op: WarpReduceOp, input_var: impl Into<String>, output_var: impl Into<String>) -> Self {
        Self {
            op,
            input_var: input_var.into(),
            output_var: output_var.into(),
            simd_width: 32,
            dtype: "float".to_string(),
        }
    }

    /// Create a sum reduction.
    pub fn sum(input_var: impl Into<String>, output_var: impl Into<String>) -> Self {
        Self::new(WarpReduceOp::Sum, input_var, output_var)
    }

    /// Create a max reduction.
    pub fn max(input_var: impl Into<String>, output_var: impl Into<String>) -> Self {
        Self::new(WarpReduceOp::Max, input_var, output_var)
    }

    /// Configure SIMD width.
    pub fn with_simd_width(mut self, width: u32) -> Self {
        self.simd_width = width;
        self
    }

    /// Configure data type.
    pub fn with_dtype(mut self, dtype: &str) -> Self {
        self.dtype = dtype.to_string();
        self
    }
}

impl Stage for WarpReduceStage {
    fn includes(&self) -> Vec<&'static str> {
        vec![]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![]
    }

    fn struct_defs(&self) -> String {
        // No additional struct defs needed - we inline the shuffle code
        String::new()
    }

    fn emit(&self, _prev: &str) -> (String, String) {
        let mut code = String::new();
        code.push_str(&format!("    // Warp-level SIMD {:?} reduction\n", self.op));

        if force_simd_sum_reduce() {
            let func = match self.op {
                WarpReduceOp::Sum => "simd_sum",
                WarpReduceOp::Max => "simd_max",
            };
            code.push_str(&format!(
                "    {dtype} {out} = {func}({in_var});",
                dtype = self.dtype,
                out = self.output_var,
                func = func,
                in_var = self.input_var
            ));
            return (self.output_var.clone(), code);
        }

        // Prefer explicit shuffle-xor reductions (what this stage is documented to do) to better
        // match the monolithic Context kernels and avoid backend-dependent lowering of `simd_sum`.
        //
        // Fall back to `simd_sum/simd_max` if the configured SIMD width is unexpected.
        let simd_width = self.simd_width;
        let is_pow2 = simd_width.is_power_of_two();
        let supported = is_pow2 && (2..=64).contains(&simd_width);

        if supported {
            code.push_str(&format!(
                "    {dtype} {out} = {in_var};\n",
                dtype = self.dtype,
                out = self.output_var,
                in_var = self.input_var
            ));

            // Emit a fixed, compile-time reduction tree for the common 32-wide SIMD groups.
            // (Apple GPUs are 32; keeping this unrolled helps the Metal compiler.)
            if simd_width == 32 {
                match self.op {
                    WarpReduceOp::Sum => {
                        for offset in [16u32, 8, 4, 2, 1] {
                            code.push_str(&format!(
                                "    {out} += simd_shuffle_xor({out}, (ushort){off});\n",
                                out = self.output_var,
                                off = offset
                            ));
                        }
                    }
                    WarpReduceOp::Max => {
                        for offset in [16u32, 8, 4, 2, 1] {
                            code.push_str(&format!(
                                "    {out} = max({out}, simd_shuffle_xor({out}, (ushort){off}));\n",
                                out = self.output_var,
                                off = offset
                            ));
                        }
                    }
                }
            } else {
                // Generic path: let Metal decide if it can unroll.
                code.push_str(&format!(
                    "    for (uint off = {simd_width}u >> 1; off > 0; off >>= 1) {{\n",
                    simd_width = simd_width
                ));
                match self.op {
                    WarpReduceOp::Sum => {
                        code.push_str(&format!(
                            "        {out} += simd_shuffle_xor({out}, (ushort)off);\n",
                            out = self.output_var
                        ));
                    }
                    WarpReduceOp::Max => {
                        code.push_str(&format!(
                            "        {out} = max({out}, simd_shuffle_xor({out}, (ushort)off));\n",
                            out = self.output_var
                        ));
                    }
                }
                code.push_str("    }\n");
            }
        } else {
            let func = match self.op {
                WarpReduceOp::Sum => "simd_sum",
                WarpReduceOp::Max => "simd_max",
            };
            code.push_str(&format!(
                "    {dtype} {out} = {func}({in_var});",
                dtype = self.dtype,
                out = self.output_var,
                func = func,
                in_var = self.input_var
            ));
        }

        (self.output_var.clone(), code)
    }
}
