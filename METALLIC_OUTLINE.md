# METALLIC Module Documentation

This document outlines all the primitives and modules that the metallic module implements and what it currently supports.

## Core Primitives

### metallic::Tensor
A zero-copy tensor backed by a retained Metal buffer. Provides views into f32 contents with GPU synchronization support.

### metallic::Context
The main context for Metal operations, containing the device, command queue, memory pool, and cached compute pipelines.

### metallic::Operation
A trait for GPU operations that can encode themselves into Metal command buffers.

### metallic::CommandBuffer
A wrapper around Metal command buffers with a simple API to record high-level operations.

### metallic::MetalError
Error type for all Metal-related operations in the module.

## Core Modules

### metallic::cacheable
Provides the `Cacheable` trait for types that can be stored in and retrieved from the resource cache using unique keys.

### metallic::cache_keys
Defines key structures for caching Metal resources:
- `MpsGemmKey` - For MPS matrix multiplication operations
- `MpsMatrixDescriptorKey` - For MPS matrix descriptors
- `SdpaKey` - For Scaled Dot Product Attention operations

### metallic::cacheable_resources
Cacheable wrappers for Metal Performance Shader resources:
- `CacheableMpsGemm` - Wrapper for MPSMatrixMultiplication
- `CacheableMpsMatrixDescriptor` - Wrapper for MPSMatrixDescriptor

### metallic::cacheable_sdpa
Cacheable wrapper for Scaled Dot Product Attention operations.

### metallic::context
Implements the main `Context` struct for managing Metal resources and operations.

### metallic::encoder
Utility functions for encoding Metal operations:
- `set_compute_pipeline_state`
- `set_buffer`
- `set_bytes`
- `dispatch_threadgroups`

### metallic::error
Defines the `MetalError` enum for all Metal-related errors.

### metallic::matmul
Matrix multiplication operations using Metal Performance Shaders:
- `mps_matrix_from_buffer` - Create MPSMatrix from MTLBuffer
- `encode_mps_matrix_multiplication` - Encode matrix multiplication
- `MatMulOperation` - High-level matmul operation

### metallic::operation
Defines the `Operation` trait and `CommandBuffer` wrapper.

### metallic::pool
Memory management with a simple bump-allocator for Metal buffers:
- `MemoryPool` - Fixed-size memory pool for tensor allocation

### metallic::resource_cache
Generic resource cache using FxHashMap for high-performance in-memory key-value storage:
- `ResourceCache` - Caches GEMM operations, matrix descriptors, and SDPA operations
- `CacheStats` - Statistics about cache usage

### metallic::scaled_dot_product_attention
Implementation of Scaled Dot Product Attention (SDPA) using Metal:
- `scaled_dot_product_attention_impl` - Standalone SDPA implementation

### metallic::softmax
Fused softmax implementation with causal masking support:
- `ensure_fused_softmax_pipeline` - Compile and cache the softmax pipeline
- `SoftmaxOperation` - Operation that runs the fused softmax kernel

### metallic::tensor
Tensor implementation with various creation and manipulation methods:
- `create_tensor_from_slice` - Create tensor from host data
- `create_tensor` - Create uninitialized tensor
- `zeros`, `ones`, `from_vec`, `arange` - Tensor creation helpers
- Element-wise operations (add, sub, mul, div)
- Shape manipulation (reshape, flatten)

### metallic::layernorm
Layer normalization implementation:
- `ensure_layernorm_pipeline` - Compile and cache the layer norm pipeline
- `LayerNorm` - Operation that runs layer normalization

### metallic::gelu
GELU activation function implementation:
- `ensure_gelu_pipeline` - Compile and cache the GELU pipeline
- `Gelu` - Operation that runs GELU activation

### metallic::model
Model-related functionality (details not provided in available files).

### metallic::tokenizer
Tokenizer implementation for processing text (details not provided in available files).