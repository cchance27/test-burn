# Metallic Dispatcher Documentation

This document details the matmul dispatcher and softmax parameterization features of the Metallic framework.

## Matmul Dispatcher

The matmul dispatcher provides shape-aware selection of optimal matmul implementations based on matrix dimensions, device capabilities, and environment preferences.

### Design Principles

- **Zero-copy tensor handling**: All tensor views preserve shape and stride metadata to avoid unnecessary copies
- **Strongly-typed selection**: Uses exhaustive enums to prevent invalid backend/variant combinations
- **Tunable heuristics**: Thresholds can be adjusted via environment variables for A/B testing

### Selection Strategy

The dispatcher follows this selection order:

1. **Environment Override**: If `METALLIC_MATMUL_BACKEND` is set to a specific backend, use that
2. **Small-N Path**: If N dimension ≤ `METALLIC_MATMUL_SMALLN_MAX_N` (default 8), consider SmallN path
3. **SIMD-Optimized GEMM**: If device supports SIMD matrix multiply and M ≥ `METALLIC_MATMUL_SIMD_M_MIN` and N ≥ `METALLIC_MATMUL_SIMD_N_MIN`, use SIMD path
4. **Fallback**: Use MLX GEMM implementation

### Public API

The dispatcher is accessible through the `MatmulDispatchOp`:

```rust
use metallic::kernels::matmul_dispatcher::MatmulDispatchOp;

// This will route through the dispatcher
let result = ctx.call::<MatmulDispatchOp>((&a, &b, &c, trans_a, trans_b, alpha, beta))?;
```

## Softmax Parameterization

The softmax implementation provides adaptive selection based on sequence length and device characteristics.

### Design Principles

- **Parameterized threadgroup sizing**: Threadgroup size adapts to sequence length (nearest power of 2, bounded by device limits)
- **Vector vs Block variants**: Short sequences use vectorized approach, long sequences use block approach
- **SIMDgroup reductions**: Optional simdgroup reduction path for improved performance
- **Defensive fallbacks**: When a kernel does not yet support features (e.g., causal masks), the dispatcher must fall back to the legacy implementation instead of emitting incorrect results

> **Current limitation:** `vec-softmax` is only enabled for non-causal rows with zero `query_offset` in FP16. The dispatcher automatically routes other cases back to the legacy softmax until masking support is implemented. Developers adding new kernels must explicitly gate unsupported modes to avoid silent accuracy bugs.

## Environment Variables

### Matmul Dispatch Variables

- `METALLIC_MATMUL_BACKEND`: Forces a specific matmul backend implementation for testing
  - `auto` (default): Use dispatcher to select optimal backend
  - `mlx`: Force MLX-based implementation
  - `mps`: Force MPS-based implementation 
  - `gemv` or `gemv`: Force legacy GEMV implementation

- `METALLIC_MATMUL_SMALLN_MAX_N`: Maximum N dimension for small-N optimization (default: 8)

- `METALLIC_MATMUL_SIMD_M_MIN`: Minimum M dimension to enable SIMD-optimized GEMM (default: 64)

- `METALLIC_MATMUL_SIMD_N_MIN`: Minimum N dimension to enable SIMD-optimized GEMM (default: 16)

- `METALLIC_MATMUL_FORCE_SMALLN`: Force the dispatcher to use small-N optimization path when set to a truthy value

### Softmax Variables

- `METALLIC_SOFTMAX_BACKEND`: Forces a specific softmax backend implementation
  - `auto` (default): Select based on sequence length
  - `vec`: Force vectorized implementation (for short sequences)
  - `block`: Force block implementation (for long sequences)

### Instrumentation Variables

- `METALLIC_LOG_LEVEL`: Sets the log level for instrumentation

- `METALLIC_METRICS_JSONL_PATH`: Path where metrics should be persisted as JSONL

- `METALLIC_METRICS_CONSOLE`: Enables console metrics emission

- `METALLIC_ENABLE_PROFILING`: Enables detailed profiling instrumentation

## Feature Flags

- `exp_kernels`: Enables experimental kernels for A/B testing

## Cache Key Extensions

The kernel cache now includes additional specialization factors:
- Transpose flags (for A/B matrices)
- Beta ≠ 0 marker (for fused alpha/beta operations) 
- Small-N bucket identifiers
- Sequence length buckets for softmax
- Causal flag for attention operations

## A/B Testing Protocol

To run A/B tests comparing different implementations:

```bash
# Test MLX vs MPS for matmul
METALLIC_MATMUL_BACKEND=mlx cargo bench matmul_dispatcher
METALLIC_MATMUL_BACKEND=mps cargo bench matmul_dispatcher

# Test different small-N thresholds
METALLIC_MATMUL_SMALLN_MAX_N=4 cargo bench matmul_dispatcher_smalln
METALLIC_MATMUL_SMALLN_MAX_N=8 cargo bench matmul_dispatcher_smalln
METALLIC_MATMUL_SMALLN_MAX_N=16 cargo bench matmul_dispatcher_smalln
```

## Benchmarking

The framework includes several benchmark suites:

- `matmul_dispatcher_bench.rs`: Tests dispatcher behavior and small-N cases
- `softmax_dispatcher_bench.rs`: Tests softmax parameterization
- `mlx_vs_mps_matmul_benchmark.rs`: Compares MLX vs MPS implementations
- `sdpa_benchmark.rs`: Tests scaled dot product attention
