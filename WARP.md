# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

The Metallic Framework is an experimental Apple Metal-based framework for AI model inference optimization. The project prioritizes **performance and latency** above all else, followed by memory usage and developer experience. It uses custom GGUF reading via mmap and focuses exclusively on Apple Metal, avoiding higher-level frameworks like burn or candle for direct control.

**Current Status**: Early development phase, targeting Qwen2.5 models (0.5B and 3B) with performance at ~56tok/s compared to LLamaStudio's 150tok/s benchmark.

## Common Development Commands

### Building and Testing
- **Build entire workspace**: `cargo build`
- **Format code**: `cargo fmt`
- **Lint and auto-fix**: `cargo clippy --fix --allow-dirty --allow-staged`
- **Run tests**: `cargo test`
- **Run benchmarks**: `cargo bench --bench <benchmark_name>` (e.g., `cargo bench --bench tokenizer_benchmark`)
- **Pre-commit workflow**: `cargo fmt && cargo clippy --fix --allow-dirty --allow-staged && cargo build`

### Individual Crates
The project uses a workspace structure with multiple crates:
- **Core library**: `cargo build -p metallic`
- **CLI tool**: `cargo build -p metallic_cli`
- **Environment utilities**: `cargo build -p metallic_env`
- **Instrumentation**: `cargo build -p metallic_instrumentation`
- **CLI helpers**: `cargo build -p metallic_cli_helpers`

### Running the CLI
- **Basic inference**: `cargo run -- --gguf-path path/to/model.gguf --prompt "Your prompt here"`
- **TUI mode** (default): Interactive terminal interface with metrics
- **Text mode**: `cargo run -- --output-format text --gguf-path path/to/model.gguf --prompt "Your prompt"`
- **JSON mode**: `cargo run -- --output-format json --gguf-path path/to/model.gguf --prompt "Your prompt"`

### Benchmarks
Available benchmarks (all use Criterion):
- `tokenizer_benchmark` - Tokenizer performance
- `sdpa_benchmark` - Scaled Dot-Product Attention
- `sdpa_variant_benchmark` - SDPA variants comparison
- `tensor_benchmark` - Tensor operations
- `gguf_quant_benchmark` - GGUF quantization
- `swiglu_cache_benchmark` - SwiGLU activation caching
- `softmax_backend_benchmark` - Softmax implementations
- `mlx_vs_mps_matmul_benchmark` - MLX vs MPS matrix multiplication

## Architecture Overview

### Workspace Structure
- **`src/`**: Main CLI application with TUI, text, and JSON output modes
- **`crates/metallic/`**: Core framework library containing:
  - **Context management**: GPU resource allocation and synchronization
  - **GGUF loading**: Memory-mapped model file reading and parsing
  - **Tensor operations**: Custom Metal compute kernels
  - **Model inference**: Qwen2.5 transformer implementation
  - **Tokenization**: GGUF-compatible tokenizer
  - **Generation**: Streaming token generation with configurable sampling
- **`crates/metallic_env/`**: Environment variable management with type safety
- **`crates/metallic_instrumentation/`**: Performance profiling and metrics collection
- **`crates/metallic_cli_helpers/`**: Shared CLI utilities and event handling

### Key Components

**Context (`Context<T>`)**:
- Central resource manager for Metal devices, command queues, and buffers
- Handles tensor lifecycle and GPU synchronization (`context.synchronize()`)
- Provides caching for compiled Metal kernels and compute pipelines
- Supports both MPS and MLX matrix multiplication backends via `FORCE_MATMUL_BACKEND` env var

**GGUF Integration**:
- Memory-mapped file reading for efficient model loading
- Custom quantization support (Q8_0 format)
- Metadata parsing for model configuration and tokenizer setup

**Metal Kernels**:
- Custom compute shaders in `kernels/` directory
- Performance-critical operations like SDPA, SwiGLU, RoPE, layer normalization
- MLX-based GEMM implementation with zero-copy batching

**Instrumentation System**:
- GPU operation timing and memory tracking
- Hierarchical metric collection with exporters
- TUI integration for real-time performance monitoring

### Performance Architecture
- **Zero-copy abstractions**: Minimize cloning, use memory-mapped files
- **Metal-native**: Direct Metal API usage for maximum performance
- **Caching strategies**: Compiled kernels, tensor resources, and compute pipelines
- **Streaming generation**: Token-by-token output with backpressure handling
- **Defensive programming**: Early failures with strongly-typed errors

## Development Guidelines

### Code Quality Requirements
- **Performance first**: All decisions prioritize performance and latency
- **Strongly typed errors**: Avoid `unwrap()`/`expect()` outside tests
- **Comprehensive testing**: All major features need tests in `metallic::tests`
- **No placeholders**: Never leave `todo!()` or placeholder code in completed work
- **Idiomatic Rust**: Use proper error handling, exhaustive pattern matching
- **Zero-copy when possible**: Prefer references over cloning large data structures

### Metal-Specific Considerations
- **GPU synchronization**: Always call `context.synchronize()` when tensors need to be settled
- **Apple Silicon required**: All GPU operations require Metal-capable hardware
- **Backend switching**: Use `FORCE_MATMUL_BACKEND=mlx|mps` to control matrix multiplication backend
- **Pipeline caching**: Leverage context-owned caches to avoid recompilation costs

### Testing Philosophy
- **Semantic validation**: Test for correctness, not just compilation
- **Extreme value testing**: Test edge cases, under/over-runs
- **Strict tolerances**: Don't adjust tests to work around issues; fix the underlying problems
- **Comparison against reference**: Use PyTorch/NumPy for expected outputs, burn-rs for complex comparisons
- **GPU hardware required**: Metal tests must run on Apple hardware

### Environment Variables
The framework uses typed environment variables for configuration:
- `METALLIC_LOG_LEVEL`: Tracing verbosity
- `METALLIC_METRICS_CONSOLE`: Enable console metrics output
- `METALLIC_METRICS_JSONL_PATH`: JSONL metrics export path
- `FORCE_MATMUL_BACKEND`: Choose `mlx` or `mps` for matrix operations

Use the `metallic_env` helpers for type-safe environment variable access.

### Tech Debt Management
- Mark technical debt with `DEBT:` comments
- Prefer fast primitives (e.g., `FxHashMap` over `HashMap`)
- Update related comments when changing referenced code
- Use enums with exhaustive matching to prevent runtime errors

### Dependency Management
- **DO NOT** manually edit `Cargo.toml` or `Cargo.lock`
- Use `cargo add`/`cargo remove` for dependency changes
- Prefer actively maintained, non-framework crates
- Avoid high-level ML frameworks (burn, candle) to maintain control

## Model Support

Currently supports **Qwen2.5** models:
- Architecture: Transformer with RoPE, SwiGLU, RMSNorm
- Formats: F16 and Q8_0 quantization
- Context: Configurable context length
- Sampling: Temperature, top-p, top-k with streaming output

Models should be placed in `models/` directory (gitignored).

## Documentation

Key documentation files:
- `docs/GGUF_SPEC.md`: GGUF format specification
- `docs/MLX_MATMUL.md`: MLX backend architecture and testing
- `docs/KERNELS.md`: Metal kernel implementation details
- `docs/TENSOR_SYNC.md`: GPU synchronization patterns
- `AGENTS.md`: Project rules and development guidelines

## Python Integration

PyTorch comparison scripts in `crates/metallic/pytorch/`:
- `compare.py`: Output validation against reference implementations
- `benchmark.py`: Performance comparison tooling
- `instrument_qwen25.py`: Model introspection and debugging

These scripts generate expected outputs for Rust tests and validate numerical correctness.