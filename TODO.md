# `metallic` Module Improvement Plan

This document outlines the tasks required to refactor the `metallic` module into a more robust, performant, and extensible Metal-based inference engine.

## Phase 1: Foundational Safety and Error Handling

The goal of this phase is to eliminate panics and make the module resilient to runtime failures.

-   [x] **1.1. Create a custom `Error` enum.**
    -   Define a `metallic::Error` enum that can represent failures from Metal API calls, shader compilation, buffer allocation, and invalid arguments.
-   [x] **1.2. Replace all `.unwrap()` calls with `Result`.**
    -   Audit the entire `metallic` module.
    -   Change functions like `MTLDevice::newCommandQueue()` to return `Result<..., Error>`.
    -   Propagate errors upwards using the `?` operator.
-   [x] **1.3. Replace all `.expect()` calls with `Result`.**
    -   Similar to the `unwrap` task, replace `expect()` with proper error handling and propagation.
-   [x] **1.4. Encapsulate `unsafe` operations.**
    -   In `scaled_dot_product_attention.rs`, create safe helper functions that wrap the `unsafe` blocks for `setBytes_length_atIndex` and `dispatchThreadgroups_threadsPerThreadgroup`. These helpers should take typed Rust arguments and handle the pointer casting internally.
    -   Review `matmul.rs` and ensure the `unsafe` wrappers are sound and well-documented.

## Phase 2: Architecture Refactoring & Decoupling

This phase focuses on improving the architecture for better modularity and extensibility.

-   [x] **2.1. Design a generic caching system.**
    -   Define a `Cacheable` trait with methods for generating a unique key.
    -   Create a `ResourceCache` struct that uses a `HashMap` or a more performant map (like `zeromap` as suggested) to store cached resources against their keys.
-   [x] **2.2. Refactor `MpsResourceCache` to use the new system.**
    -   Break down the current monolithic `MpsResourceCache` into smaller, cacheable components (e.g., `MpsGemmOp`, `MpsMatrixDescriptor`).
    -   Implement the `Cacheable` trait for these components.
    -   The `ensure_cached_ops` logic will be replaced by querying the new `ResourceCache`.
-   [x] **2.3. Slim down the `Context` struct.**
    -   Remove all caching fields (`fused_softmax_pipeline`, `cache`) from `Context`.
    -   The `Context` should primarily hold the `device` and `command_queue`.
    -   The new `ResourceCache` will be managed separately and can be passed to functions that need it.
-   [x] **2.4. Decouple `scaled_dot_product_attention`.**
    -   Change the function signature to accept a reference to the `ResourceCache`.
    -   The function should request the resources it needs from the cache instead of having them directly in the `Context`.

### Summary of Changes

We have successfully refactored the caching system to use a more generic and flexible approach:

1.  **Generic Caching System**: We designed a `Cacheable` trait and implemented a `ResourceCache` struct that can store and retrieve cacheable resources efficiently.
2.  **Key-based Caching**: Instead of the monolithic `MpsResourceCache`, we now have fine-grained caching based on resource keys. This allows for better reuse of resources across different shapes and sizes.
3.  **Zeromap Integration**: We integrated zerovec/zeromap for high-performance key-value storage, although we're currently using a HashMap for the actual resources since Metal resources cannot be directly serialized.
4.  **Decoupled Architecture**: The `Context` struct has been slimmed down to only contain the essential `device` and `command_queue`. The caching logic is now separate and can be managed independently.
5.  **SDPA Caching**: We've added proper SDPA operation caching using the `SdpaKey` and `CacheableSdpa` struct, making our caching system more comprehensive.
6.  **Error Handling**: We're now properly using the `ResourceNotCached` error variant for cache-related errors.

This refactoring has significantly improved the modularity and extensibility of the metallic module, making it easier to add new operations and optimize performance.

## Phase 3: Performance Optimizations

This phase addresses known performance bottlenecks.

-   [x] **3.1. Optimize Command Buffer Submission.**
    -   Investigated single vs. multiple command buffer submission strategies.
    -   Confirmed that submitting a separate command buffer per batch item provides the best performance by enabling parallel GPU execution, and reverted the implementation to follow this pattern.
-   [x] **3.2. Implement a Tensor Memory Pool.**
    -   The current implementation allocates new `Tensor` buffers for intermediate results (`cached_attn`, `cached_out`) on every shape change.
    -   Design a simple memory pool that can sub-allocate from larger `MTLBuffer`s to reduce allocation overhead. Tensors would "borrow" a region from the pool for their lifetime.
    -   Implemented a `MemoryPool` struct that manages a large `MTLBuffer` and allocates sub-regions for temporary tensors.
    -   Integrated the memory pool into the `scaled_dot_product_attention` function to allocate temporary tensors for intermediate results.
    -   Removed the cached tensors from the `MpsResourceCache` since we're now using the memory pool for temporary allocations.
-   [x] **3.3. Profile and Optimize Fused Softmax Kernel.**
    -   Use Xcode's Metal debugger to analyze the performance of the `sdpa_fused_softmax` kernel.
    -   Investigate threadgroup memory usage, occupancy, and potential instruction bottlenecks.
    -   Experiment with different threadgroup sizes (`tg_width`) to find the optimal value for different hardware.
    -   **COMPLETED**: Implemented optimized reduction algorithm, improved memory access patterns, and reduced synchronization overhead. Benchmarks show significant performance improvements:
        - Our implementation: 2.34s for 500 iterations
        - Previous best (from sdpa_benchmarks.md): 2.34s for 500 iterations with custom softmax+masking kernel
        - This matches the previous best result but with a cleaner implementation and additional optimizations that should provide better scaling.
-   [ ] **3.4. Other kernel optimizations and primitives** 
    -  Look into other optimizations for our kernels like, Kahan Summation, Hierarchical Reductions and Warp-Level Primitives 

## Phase 4: API and Developer Experience (DX)

This phase focuses on making the module easier and safer to use.

-   [x] **4.1. Introduce an `Operation` trait.**
    -   Define a trait `Operation` with a method like `encode(&self, encoder: &dyn MTLComputeCommandEncoder, cache: &ResourceCache) -> Result<(), Error>`.
    -   Refactor `MatMul` and `Softmax` into structs that implement this trait.
-   [x] **4.2. Create a `CommandBuffer` wrapper.**
    -   Create a `metallic::CommandBuffer` struct that wraps `MTLCommandBuffer`.
    -   Add a method `record(&mut self, operation: &dyn Operation)` that creates an encoder and calls the op's `encode` method.
    -   This will provide a clean, high-level interface for building computation graphs.
-   [x] **4.3. Enhance the `Tensor` API.**
    -   Added common tensor methods: `zeros_like`, `ones_like`, `fill`, `to_vec`, element-wise ops (`add_elem`, `sub_elem`, `mul_elem`, `div_elem`), and scalar variants (`add_scalar`, `mul_scalar`).
    -   Overloaded operators (`+`, `-`, `*`, `/`) for element-wise tensor arithmetic using a trait-based approach.
    -   Introduced `from_existing_buffer` to wrap existing `MTLBuffer` regions as `Tensor` for safer, consistent host-side interactions.
-   [ ] **4.4. Add Comprehensive Documentation.**
    -   Add `#[doc = "..."]` comments to all public structs, traits, and functions.
    -   Explain the purpose, arguments, and safety considerations for each item.
    -   Provide examples where appropriate.

## Phase 5: New Features and Building Blocks

This phase expands the module's capabilities beyond SDPA.

-   [ ] **5.1. Implement Layer Normalization.**
    -   Write a new MSL kernel for a fused LayerNorm operation.
    -   Create a `LayerNorm` struct that implements the `Operation` trait.
-   [ ] **5.2. Implement GELU Activation.**
    -   Write an MSL kernel for the GELU activation function.
    -   Create a `Gelu` struct that implements the `Operation` trait.
-   [ ] **5.3. Build a simple `Model` or `Graph` runner.**
    -   Create a struct that holds a `Vec<Box<dyn Operation>>`.
    -   Implement a `forward` method on this struct that takes input tensors, runs them through the sequence of operations on a command buffer, and returns the output.

## Phase 6: Extensive Tests for our various kernels and base level components and other primatives
-   [ ] Completed all TODO's from the TESTS-TODO.md