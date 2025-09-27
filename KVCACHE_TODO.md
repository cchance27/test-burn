### TODO: High-Performance KV Cache Implementation

#### Phase 1: Pre-computation and Upfront Allocation
*Goal: Allocate all necessary memory once and pre-compute expensive, reusable data before the generation loop begins.*

-   [ ] **1.1. Centralize and Pre-allocate KV Cache Tensors:**
    -   **Action:** In the main generation function, before the loop starts, iterate from `layer = 0` to `n_layers - 1` and use the existing `ctx.alloc_kv_cache()` to pre-allocate the Key and Value tensors for every transformer block.
    -   **Rationale:** This allocates all required GPU memory for the full context length (`max_seq_len`) upfront. This prevents any further large-scale allocations during the token-by-token generation, leading to stable, predictable memory usage and eliminating allocation overhead from the critical path.

-   [ ] **1.2. Pre-compute and Cache RoPE Frequencies:**
    -   **Action:** In the `Qwen25` model struct, create and compute two new `Tensor`s at initialization: `rope_cos_cache` and `rope_sin_cache`. These will store the rotary embedding values for every position from `0` to `max_seq_len`.
    -   **Rationale:** The current implementation recalculates `sin` and `cos` for all sequence positions on every single step. This is highly redundant. Pre-computing them once removes this expensive trigonometric work from the generation loop entirely.

#### Phase 2: Implement Incremental Forward Pass
*Goal: Create a new `forward_step` function that processes only a single token at a time, using the pre-allocated caches.*

-   [ ] **2.1. Create a new `forward_step` Function:**
    -   **Action:** Implement a new method in `Qwen25`: `fn forward_step(&self, token_embedding: &Tensor, current_pos: usize, ctx: &mut Context) -> Result<Tensor, MetalError>`.
    -   **Parameters:**
        -   `token_embedding`: The embedding for the single new token, with shape `[1, 1, d_model]`.
        -   `current_pos`: The current index in the sequence we are generating for.
    -   **Rationale:** This function will be the core of the incremental generation, replacing the need to process the full context on each iteration.

-   [ ] **2.2. Update Transformer Block Logic for a Single Step:**
    -   **Q, K, V Projection:** Inside `forward_step`, when passing through each `TransformerBlock`, the Q, K, and V matrices are calculated only for the single input token embedding. This keeps these intermediate tensors small and constant-sized.
    -   **RoPE from Cache:** Apply RoPE to the single-step Q and K heads by taking a 1-dimensional slice from the `rope_cos_cache` and `rope_sin_cache` at index `current_pos`. This is a simple, fast lookup.
    -   **In-Place KV Cache Update:** Use the existing `ctx.write_kv_step()` function to copy the newly computed K and V heads into their respective cache tensors in `ctx.kv_caches` at the `current_pos` offset. This function uses an efficient device-to-device copy (`MTLBlitCommandEncoder`), avoiding any CPU roundtrips.
    -   **Attention with Full Cache:** For the `scaled_dot_product_attention` call:
        -   `Q`: Use the single-step query vector.
        -   `K` & `V`: Use a **zero-copy view** or **slice** of the full K and V cache tensors for the current layer, from position `0` to `current_pos`. This is the most critical optimization. We must ensure the `Tensor` struct can create a view of a buffer without copying any data on the GPU.

-   [ ] **2.3. Modify the Generation Loop:**
    -   **Action:** Update `generate_autoregressive_..._streaming` to call the new `qwen.forward_step()` in a loop for each new token.
    -   **Rationale:** This replaces the inefficient full `qwen.forward()` call, ensuring each step is a small, constant-time operation (excluding the attention calculation, whose cost will grow linearly with sequence length, which is unavoidable). The `ctx.reset_pool()` and `ctx.clear_cache()` calls should be kept to reclaim memory for temporary tensors created *within* the step.

#### Phase 3: API and Developer Experience
*Goal: Ensure the new implementation is clean, easy to use, and maintainable.*

-   [ ] **3.1. Implement Zero-Copy `Tensor::slice`:**
    -   **Action:** Add a `slice` method to the `Tensor` struct. This method should not copy any GPU data. It should create a new `Tensor` object that points to the same underlying `MTLBuffer` but with a different offset and shape/stride information.
    -   **Rationale:** This is essential for performance. Copying slices of the KV cache on every layer of every step would negate the benefits of the cache.

-   [ ] **3.2. Finalize and Document:**
    -   **Action:** Rename the old, inefficient generation function to something like `_debug_generate_full_context` to mark it as a reference/debugging tool. Make the new KV-cache-enabled function the primary entry point.
    -   **Action:** Add documentation to `forward_step` explaining the mechanism, tensor shapes, and caching strategy.

#### Phase 4: Verification
*Goal: Prove the new implementation is both correct and performant.*

-   [ ] **4.1. Numerical Correctness Check:**
    -   **Action:** Generate a sequence of ~50 tokens with both the old and new methods. Compare the final logits. They should be identical (or within a very small tolerance for floating-point error), proving the new logic is a correct optimization.
-   [ ] **4.2. Profile Performance and Memory:**
    -   **Action:** Run the new implementation and observe memory usage. It should show a high initial allocation and then remain **completely flat** for the duration of the token generation.
    -   **Action:** Benchmark tokens/second. The new method should be significantly faster, with the performance gap widening as the sequence gets longer.
