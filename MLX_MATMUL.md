# MLX MatMul Backend

This document summarizes the design of the `mlxmatmul` backend, how it integrates
with the Metallic context, and which scenarios are currently covered by tests.
It also calls out qwen25 inference paths that still need direct coverage when
comparing the MLX and MPS backends.

## High-level architecture

* **Metal shader**: `src/metallic/kernels/mlxmatmul/mlx.metal` adapts the
  experimental MLX GEMM kernel so it can honor both `alpha`/`beta` parameters
  by wiring the `use_out_source` and `do_axpby` function constants into the
  existing AddMM code path.
* **Runtime integration**: `src/metallic/kernels/mlxmatmul/mlx.rs` implements
  `MatMulMlxOp`, which is registered as a `KernelInvocable`.  The operator
  inspects operand layouts, sets up `GemmParams`/`GemmAddmmParams`, and binds the
  MPS-style `Tensor` handles to Metal buffers before encoding the dispatch.
* **Pipeline cache**: Each `Context` owns an `MlxKernelCache`.  The cache lazily
  compiles the Metal library and specializes pipelines keyed by the dispatch
  characteristics (batched vs. single, layout alignment, whether beta is used,
  etc.).  This avoids paying the compilation cost on every matmul call.
* **Context routing**: `Context::matmul` and `Context::matmul_alpha_beta`
  latch the value of the `FORCE_MATMUL_BACKEND` environment variable when the
  context is constructed.  Setting `FORCE_MATMUL_BACKEND=mlx` forces dispatches
  through `MatMulMlxOp`; leaving it unset or assigning `mps` continues to use
  the legacy MPS implementation.  Any other value falls back to MPS.  Instrumentation
  hooks record which backend executed each dispatch so we can profile mixed
  workloads.

## Layout handling

`MatMulMlxOp` accepts the same tensor views that the MPS backend does.  We
validate that each operand can be interpreted as row- or column-major with an
integer leading dimension.  Padded row/column strides (e.g. sliced tensors that
keep extra storage at the end of each row) are accepted as long as the leading
dimensions are monotonically increasing.  Batched inputs can either be dense or
views with consistent batch strides.

When `beta != 0`, we require the caller to provide the destination tensor.
`MatMulMlxOp` passes that tensor as both an input and an output buffer so the
kernel can perform the fused `C = αAB + βC` update.

## Testing coverage

### Performance note: zero-copy batching

- Unlike MPS, MLX GEMM supports arbitrary batch strides (batch_stride_a/b/d) and non-compact row/column leading dimensions (lda/ldb/ldd) directly.
- We therefore do not compact batched inputs prior to encoding MLX GEMM. Any padding between matrices in a batch is handled by the kernel via the batch stride parameters.
- Avoiding MPS-style compaction prevents an extra blit/copy and aligns the integration with the experimental benchmark setup, restoring the expected performance advantage of the MLX kernel.


* `src/metallic/kernels/matmul/mlx_test.rs` compares MLX vs. MPS for
  * baseline matmul without transpositions,
  * left/right/both transposed operands,
  * alpha/beta accumulation with non-zero beta,
  * non-contiguous (padded) views,
  * small batched workloads.
* `src/metallic/tests/matmul.rs` focuses on the qwen25 inference shapes and
  alpha/beta extremes that we exercise in practice.  The helper enforces a
  combined absolute/relative tolerance so we detect real divergences without
  flagging expected float32 rounding differences.

Because Metal execution is unavailable inside the Linux CI sandbox, all GPU
checks (`cargo build`, `cargo clippy`, `cargo test`) must be run on Apple
hardware with a Metal-capable GPU.

## Known gaps and follow-ups

The integration tests mirror every matmul call currently used by the qwen25
inference path, but there are still scenarios we should keep an eye on:

* **Cache-aware alpha/beta paths** – `Context::matmul_alpha_beta_with_cache`
  can update an existing tensor while reusing a `ResourceCache`.  qwen25 does
  not trigger this code path in the current tests, so we should add coverage if
  we begin to rely on cached dispatches in production.
* **Mixed-layout alpha/beta** – We only exercise `beta != 0` on contiguous
  inputs.  If qwen25 (or future models) perform `αAB + βC` using sliced or
  transposed operands we need to extend the tests accordingly.
* **Higher-rank batching** – The batched tests cover `[batch, M, K]` ×
  `[batch, K, N]`.  Qwen currently issues at most 3-D tensors, but if we expand
  to more complex broadcasted batches or ragged layouts we should validate the
  MLX kernel’s stride calculations under those shapes.
* **Other dtypes** – The MLX kernel is validated with `f32`.  If we introduce
  `f16`/`bf16` execution we will need new tests because the tolerances and
  kernel constant choices will differ.

Whenever we extend qwen inference to new matmul usages, mirror the scenario in
`src/metallic/tests/matmul.rs` so MLX and MPS remain bit-for-bit compatible.
