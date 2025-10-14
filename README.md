# Metallic Framework

I'm in the process of writing a HIGHLY experimental Apple Metal framework so that i can play with implementing model inference, and experimental optimization.

Sadly we're at the stage of "get it working" still, we're looking to first try to match llamacpp and pytorch performance. 

Using only actively maintained crates, non-framework (no burn, no candle, etc), custom GGUF reading via mmap.

Only supporting Apple Metal hence the name... Metallic.

Right now we're testing with Qwen2.5 (0.5b and 3b)

Qwen2.5b is testing at ~56tok/s vs lmstudio's 150tok/s

So we've got a long ways to go.

## Environment Variables

The Metallic framework supports various environment variables to control behavior, enable debugging, and facilitate A/B testing of different kernel implementations.

### Matmul Dispatch Variables

- `METALLIC_MATMUL_BACKEND`: Forces a specific matmul backend implementation for testing. Valid values:
  - `auto` (default): Use dispatcher to select optimal backend
  - `mlx`: Force MLX-based implementation
  - `mps`: Force MPS-based implementation 
  - `gemv` or `legacy_gemv`: Force legacy GEMV implementation

- `METALLIC_MATMUL_SMALLN_MAX_N`: Maximum N dimension for small-N optimization (default: 8). Matmuls with N ≤ this value may use specialized kernels.

- `METALLIC_MATMUL_SIMD_M_MIN`: Minimum M dimension to enable SIMD-optimized GEMM (default: 64).

- `METALLIC_MATMUL_SIMD_N_MIN`: Minimum N dimension to enable SIMD-optimized GEMM (default: 16).

- `METALLIC_MATMUL_FORCE_SMALLN`: Force the dispatcher to use small-N optimization path when set to a truthy value (1, true, yes, on).

### Softmax Variables

- `METALLIC_SOFTMAX_BACKEND`: Forces a specific softmax backend implementation for testing. Valid values:
  - `auto` (default): Select based on sequence length
  - `vec`: Force vectorized implementation (for short sequences)
  - `block`: Force block implementation (for long sequences)

### Instrumentation Variables

- `METALLIC_LOG_LEVEL`: Sets the log level for instrumentation (e.g., "info", "debug", "trace").

- `METALLIC_METRICS_JSONL_PATH`: Path where metrics should be persisted as JSONL.

- `METALLIC_METRICS_CONSOLE`: Enables console metrics emission when set to a truthy value.

- `METALLIC_ENABLE_PROFILING`: Enables per-command-buffer GPU latency emission and detailed profiling instrumentation (impacts performance).

## Feature Flags

- `exp_kernels`: Enables experimental kernels for testing and A/B comparison.