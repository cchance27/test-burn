# Foundry Memory Management Architecture

Foundry employs a hybrid memory management strategy designed to balance high-performance inference (minimizing runtime allocation overhead) with flexibility for dynamic operations.

## 1. Memory Architectures

Foundry distinguishes between the long-lived static graph used for inference and the dynamic pool used for ad-hoc operations.

### A. Static Execution Graph (Inference Core)
The `CompiledModel` executor relies on **Static Provisioning**. All buffers required for a forward pass are pre-calculated and allocated during the `initialize_session` phase.

- **Persistence**: Buffers persist for the lifetime of the `ModelSession`.
- **Reuse**: "Intermediate" buffers (activations like `hidden_states`, `logits`) are allocated once and reused for every token step.
- **Zero-Copy**: The execution loop handles raw `MetalBuffer` handles, avoiding wrapper overhead in the hot path.
- **Allocation**: Directly uses `MetalDevice::new_buffer` with specific storage modes.

### B. Global Memory Pool (Dynamic/Testing)
For dynamic operations (e.g., unit tests, profiling, or future dynamic layers), Foundry provides a `MemoryPool` with a high-performance bump allocator.

- **Structure**: Growable list of `MTLBuffer` chunks (default 256MB initial size).
- **Strategy**: Bump allocation (O(1)).
- **Safety**: **Generational Invalidation** (Arena Pattern).
    - The pool maintains an atomic `current_generation` counter.
    - `Pooled` tensors are stamped with their creation generation.
    - Accessors verify `tensor.generation == pool.generation` to prevent Use-After-Free bugs after a `pool.reset()`.

---

## 2. Provisioning & Capacity Planning

Foundry includes a sophisticated provisioning system (`ContextConfig`) to manage VRAM usage safely.

### Context Budgeting
Before allocating the heavy KV cache, Foundry calculates a **Memory Budget**:
- **Auto**: Defaults to using a safe percentage of the system's "Recommended Max Working Set" (typically ~70-80% of total RAM on Apple Silicon).
- **Explicit**: Can be capped via `METALLIC_KV_MEMORY_BUDGET_MB`.

If the requested model context length would exceed the budget, Foundry **clamps** the context length and logs a warning, prioritizing system stability over maximum sequence length.

### Growth Strategy (KV Cache)
The Key-Value cache uses a **Geometric Doubling** strategy to avoid reserving maximum memory at startup.
1.  **Initial**: Allocates a small baseline (e.g., 2048 tokens).
2.  **Growth**: When capacity is reached, a new buffer (2x size) is allocated.
3.  **Preservation**: A GPU `blit_copy` kernel migrates existing data to the new buffer.
4.  **Alignment**: Capacity is always aligned to **128 tokens** to ensure optimal tiling for AMX/SIMD kernels.

---

## 3. Component-Specific Memory

### Model Weights
- **Source**: Memory-mapped from GGUF files via `mmap`.
- **Loading**: Weights are materialized on-demand to minimize startup latency.
    - **Staging**: Loaded into `StorageModeShared` (Host-visible).
    - **Destination**: Blitted to `StorageModePrivate` (GPU-only) for maximum bandwidth.
- **Quantization**: Weights are stored in compressed formats (Q8_0, Q4_0) in VRAM and decompressed on-the-fly during compute.

### Intermediates (Activations)
- **Format**: Typically `F16` (Half Precision).
- **Storage**: `StorageModePrivate` (default).
- **Visibility**: Can be forced to `Shared` for debugging via `METALLIC_INTERMEDIATES_SHARED=1`.

### RoPE Caches
- **Role**: Pre-computed Cosine/Sine tables for Rotary Positional Embeddings.
- **Growth**: Resized automatically in tandem with the KV Cache.

---

## 4. Storage Modes & Bandwidth

Foundry optimizes for Apple Silicon's Unified Memory Architecture (UMA) but respects the distinct storage tiers to maximize bandwidth.

| Tensor Type | Default Mode | Rationale |
| :--- | :--- | :--- |
| **Weights** | `Private` | High reuse, static. Needs max GPU bandwidth. |
| **KV Cache** | `Private` | Read every step. High bandwidth requirement. |
| **Intermediates** | `Private` | Short-lived, written/read by GPU kernels only. |
| **Staging/Inputs** | `Shared` | Written by CPU (Token IDs), read by GPU. |
| **Outputs** | `Shared` | Written by GPU (Logits), read by CPU (Sampling). |

---

## 5. Environment Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| `METALLIC_MAX_CONTEXT_LEN` | Model Default | Hard cap on the sequence length. |
| `METALLIC_KV_MEMORY_BUDGET_MB` | Auto | Max VRAM allowed for KV cache. |
| `METALLIC_INTERMEDIATES_SHARED` | 0 | Force activations to be host-visible (debug). |
| `METALLIC_KV_CACHE_SHARED` | 0 | Force KV cache to be host-visible (debug). |
| `METALLIC_PREFILL_CHUNK_SIZE` | 32 | Batch size for prompt processing. |
| `METALLIC_FOUNDRY_DECODE_BATCH_SIZE` | 64 | Batch size for speculative decoding. |
