# Tensor Synchronization Summary

This project now ships an implicit synchronization model for Metal tensors. The goal is to keep GPU work batched on a single command buffer, only force synchronization when host code genuinely needs the results, and remove the burden of sprinkling `ctx.synchronize()` throughout model code.

## What Was Implemented
- **Stateful tensors.** Each `Tensor` owns `defining_cmd_buffer: Rc<RefCell<Option<CommandBuffer>>>`. `Tensor::ensure_ready` commits and waits the buffer on demand before returning host views (`src/metallic/tensor.rs`).
- **Shared command-buffer wrapper.** `CommandBuffer` wraps `MTLCommandBuffer` in an `Arc` and tracks `committed`/`completed` flags, giving us cheap clones plus `is_committed`/`is_completed`/`ptr_eq` helpers (`src/metallic/operation.rs:160`).
- **Context-managed batching.** `Context` holds an `active_cmd_buffer` and `active_resource_cache`. `ensure_active_cmd_buffer` refreshes that pair once a buffer has been committed or finished, first waiting it out to avoid overlapping writes (`src/metallic/context.rs:294`).
- **Lazy sync on host access.** Calls such as `as_slice`, `as_mut_slice`, and `to_vec` promote pending GPU work to completion before exposing CPU data.
- **Composite op support.** `Context::call` now detects when the active buffer was committed during nested calls (e.g. custom ops that touch host memory) and swaps in a fresh buffer/cache before recording the outer operation (`src/metallic/context.rs:95`). This closed the runtime panic seen when running Qwen25.
- **API ergonomics.** High-level tensor helpers no longer spam `ctx.synchronize()`. Synchronization happens exactly when tensors leak to host code or when the user manually calls `ctx.synchronize()`.

## Fixes and Current Limitations
- ✅ Resolved `InvalidOperation("Attempted to record on a committed command buffer")` by refreshing the active buffer after nested commits.
- ✅ Removed accidental double-borrow panics in `Tensor::ensure_ready` by ensuring the defining buffer is cleared after wait.
- ⚠️ Composite ops (e.g. `SwiGLU`, `ScaledDotProductAttention`) still perform host copies inside their pipelines, which forces the active buffer to commit mid-call and defeats batching (`src/metallic/models/qwen25/mod.rs:383`, `src/metallic/kernels/scaled_dot_product_attention/mod.rs:121`).
- ⚠️ Host-side tensor transforms such as KV head repetition and debug stats still walk slices on the CPU. They work, but every call triggers a sync, so use them sparingly on the hot path.
- ⚠️ `Tensor::ensure_ready` relies on an internal `RefCell`. Re-entrant host accesses on the same tensor from multiple threads would still panic. Today we only target single-threaded paths, but this is a future sharp edge.

## Future Optimization Ideas
1. **GPU-native composites.** Move `repeat_kv_heads`, SDPA packing, and similar routines out of CPU loops into dedicated Metal kernels to keep the active buffer alive through the whole block.
2. **Download scheduling.** Introduce an explicit `Tensor::read_async`/`Tensor::map` that queues a blit copy into staging buffers instead of blocking on `waitUntilCompleted` when a host slice is requested.
3. **Debug instrumentation.** Add counters/logging inside `Context::call` when we refresh the active buffer so we can catch accidental host syncs during profiling.
4. **Safer host access.** Replace the plain `RefCell` with `try_borrow` plus a descriptive error to make accidental nested borrows easier to diagnose, and investigate RwLock-based guards if we expand to multi-threaded decoding.
5. **Resource cache lifetime.** Consider pooling `ResourceCache` instances across command buffers to reuse heavy MPS descriptors instead of tearing them down when we refresh.

## Tensor initialization entry points
All tensor construction now flows through `Tensor::new`. Callers specify the desired shape alongside two enums:

- [`TensorStorage`] chooses the allocation target (`Dedicated` allocates a fresh `MTLBuffer`, `Pooled` pulls from the transient bump allocator and requires `&mut Context`).
- [`TensorInit`] describes how to seed the buffer (`Uninitialized`, `CopyFrom(&[f32])`, or `BorrowHost(&[f32])`).

`BorrowHost` remains dedicated-only because the pooled allocator always materializes device memory before staging. Helpers such as `Tensor::zeros` and the memory pool itself are thin wrappers over this API, so future storage modes can extend `TensorStorage`/`TensorInit` without rewriting every call site.

## Validation Checklist
- `cargo test --release` on a Metal-capable Mac (many suites require GPU access).
- `cargo run --release -- ./models/qwen2.5-coder-0.5b-instruct-fp16.gguf "<prompt>"` end-to-end run – ensures lazy sync works under real model load.
- Optional: profile the number of times `Context::call` refreshes the buffer when running long prompts; if the count is high, investigate callers that touch `as_slice` or `to_vec` in tight loops.

Keep this file updated as we tighten GPU residency, add async readback, or change how command buffers are recycled.
