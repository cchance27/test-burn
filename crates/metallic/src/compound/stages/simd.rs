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
        // Generate shuffle distances dynamically based on simd_width
        // For simd_width=32: distances are 16, 8, 4, 2, 1
        // For simd_width=64: distances are 32, 16, 8, 4, 2, 1
        let mut code = format!(
            "    // Warp-level SIMD {:?} reduction (simd_width={})\n    {} {} = {};\n",
            self.op, self.simd_width, self.dtype, self.output_var, self.input_var
        );

        let op_fn = match self.op {
            WarpReduceOp::Sum => "+=",
            WarpReduceOp::Max => "= max({out}, ",
        };

        let mut distance = self.simd_width / 2;
        while distance >= 1 {
            match self.op {
                WarpReduceOp::Sum => {
                    code.push_str(&format!(
                        "    {} {} simd_shuffle_xor({}, {}u);\n",
                        self.output_var, op_fn, self.output_var, distance
                    ));
                }
                WarpReduceOp::Max => {
                    code.push_str(&format!(
                        "    {} = max({}, simd_shuffle_xor({}, {}u));\n",
                        self.output_var, self.output_var, self.output_var, distance
                    ));
                }
            }
            distance /= 2;
        }

        (self.output_var.clone(), code)
    }
}
