# Qwen2.5 Token Generation Infrastructure Overview

Based on Qwen2.5-Coder-0.5B: d_model=896, ff_dim=4864, n_heads=14, n_kv_heads=2, n_layers=24, vocab_size=151936, seq_len=32768, rope_freq_base=1000000.0, rms_eps=1e-6

## Current Token Generation Path

### 1. Token Embedding
- **Input**: [batch, seq] token IDs (uint32)
- **Operation**: Look up in embedding weight matrix
- **Tensor**: `token_embd.weight` [896, 151936] (F16)
- **Output**: [batch, seq, 896] (d_model=896)
- **Kernel Used**: CPU-based lookup (memcpy operations)
- **CPU/GPU Work**: CPU operation - no GPU kernel involved
- **Sync Points**: None

### 2. Transformer Block Processing (Repeated for each of 24 layers)
#### A. Attention Block
- **RMSNorm**: Normalize input before attention
  - **Input**: [batch, seq, 896]
  - **Tensor**: `blk.{i}.attn_norm.weight` [896] (F32)
  - **Kernel Used**: `RMSNormOp` 
  - **CPU/GPU Work**: GPU operation
  - **Sync Points**: None

- **QKV Projection**: Three GEMMs (Query, Key, Value) using fused projection
  - **Input**: [batch*seq, 896] 
  - **Weights**: 
    - Q: `blk.{i}.attn_q.weight` [896, 896] (F16)
    - K: `blk.{i}.attn_k.weight` [896, 128] (F16) 
    - V: `blk.{i}.attn_v.weight` [896, 128] (F16)
  - **Bias**: 
    - Q: `blk.{i}.attn_q.bias` [896] (F32)
    - K: `blk.{i}.attn_k.bias` [128] (F32)
    - V: `blk.{i}.attn_v.bias` [128] (F32)
  - **Output Shapes**:
    - Q: [batch*seq, 896] → [batch, seq, 14, 64] (14 heads * 64 dim = 896)
    - K: [batch*seq, 128] → [batch, seq, 2, 64] (2 KV heads * 64 dim = 128)
    - V: [batch*seq, 128] → [batch, seq, 2, 64]
  - **Kernel Used**: `MatmulDispatchOp` (selects MLX/MPS/GEMV based on shape)
    - For Q (896x896): Likely SIMD GEMM or MLX
    - For K/V (896x128): Likely MLX or Small-N GEMV optimization 
  - **Tensor Sizes**:
    - Q projection: 896×896 = 802,816 F16 parameters (~1.6M bytes)
    - K projection: 896×128 = 114,688 F16 parameters (~230K bytes)  
    - V projection: 896×128 = 114,688 F16 parameters (~230K bytes)
  - **CPU/GPU Work**: GPU operations with matmul dispatcher selecting optimal backend
  - **Sync Points**: None for async operations

- **Head Rearrangement**: Reshape Q/K/V into multi-head format
  - **Operation**: Reshape and permute tensors to [batch, heads, seq, head_dim]
  - **Q**: [batch, seq, 896] → [batch, 14, seq, 64]
  - **K**: [batch, seq, 128] → [batch, 2, seq, 64] 
  - **V**: [batch, seq, 128] → [batch, 2, seq, 64]
  - **Kernel Used**: `KvRearrangeOp`
  - **CPU/GPU Work**: GPU operation
  - **Sync Points**: None

- **RoPE Application**: Apply rotary positional encoding using cached cos/sin values
  - **Input**: Q [batch, 14, seq, 64], K [batch, 2, seq, 64]
  - **Caches**: `rope_cos_cache` [32768, 32] and `rope_sin_cache` [32768, 32] (precomputed)
  - **Kernel Used**: `RoPEOp`
  - **CPU/GPU Work**: GPU operation
  - **Sync Points**: None

- **KV Caching**: Store K/V values in KV cache for future tokens 
  - **Operation**: Write new K/V values to persistent cache
  - **Cache Size**: 24 layers × 2 (K+V) × batch × (seq+1) × head_dim
  - **Kernel Used**: Memory write operations
  - **CPU/GPU Work**: GPU (blit encoder copy operations)
  - **Sync Points**: None

- **GQA (Grouped Query Attention)**: Repeat K/V heads to match Q head count
  - **Input**: K [batch, 2, seq, 64], V [batch, 2, seq, 64]
  - **Output**: K [batch, 14, seq, 64], V [batch, 14, seq, 64] (repeated 7x)
  - **Kernel Used**: `RepeatKvHeadsOp`
  - **CPU/GPU Work**: GPU operation
  - **Sync Points**: None

- **SDPA (Scaled Dot Product Attention)**: 
  - **QK^T matmul**: [batch, 14, seq, 64] × [batch, 14, seq, 64]ᵀ → [batch, 14, seq, seq]
    - **Shape**: For single token generation: [1, 14, 1, 64] × [1, 14, cur_pos, 64]ᵀ → [1, 14, 1, cur_pos]
    - **Kernel Used**: `MatmulDispatchOp` (likely MLX for small-N case)
    - **Tensor Size**: For first token: 14×1×cur_pos (varies with context)
  - **Softmax**: Apply causal mask and softmax to attention weights
    - **Input**: [batch, 14, 1, cur_pos] 
    - **Kernel Used**: Softmax dispatcher selects vec-softmax (≤1024) or block-softmax (>1024)
    - **Tensor Size**: 14×cur_pos (e.g., 14×10 for 10 tokens in context)
  - **AV matmul**: [batch, 14, 1, cur_pos] × [batch, 14, cur_pos, 64] → [batch, 14, 1, 64]
    - **Kernel Used**: `MatmulDispatchOp`
    - **Output**: [batch, 1, 896] after reshaping
  - **CPU/GPU Work**: Three sequential GPU operations with no sync points between
  - **Sync Points**: None (all operations stay on GPU)

- **Attention Output**: Final matmul to project attention output back to d_model
  - **Input**: [batch, seq, 896] (from attention)
  - **Weight**: `blk.{i}.attn_output.weight` [896, 896] (F16)
    - **Size**: 896×896 = 802,816 F16 parameters (~1.6M bytes)
  - **Operation**: [batch*seq, 896] × [896, 896] → [batch*seq, 896]
  - **Kernel Used**: `MatmulDispatchOp` (likely SIMD GEMM)
  - **CPU/GPU Work**: GPU operation
  - **Sync Points**: None

#### B. MLP Block
- **RMSNorm**: Normalize input before MLP
  - **Input**: [batch, seq, 896] (post-attention residual)
  - **Tensor**: `blk.{i}.ffn_norm.weight` [896] (F32)
  - **Kernel Used**: `RMSNormOp`
  - **CPU/GPU Work**: GPU operation
  - **Sync Points**: None

- **SwiGLU FFN**: Feed-forward network using SwiGLU activation
  - **Gate Projection**: [batch*seq, 896] × [896, 4864] → [batch*seq, 4864]
    - **Weight**: `blk.{i}.ffn_gate.weight` [896, 4864] (F16)
    - **Size**: 896×4864 = 4,358,144 F16 parameters (~8.7M bytes)
    - **Kernel Used**: `MatmulDispatchOp` (MLX for large N=4864)
  - **Up Projection**: [batch*seq, 896] × [896, 4864] → [batch*seq, 4864]
    - **Weight**: `blk.{i}.ffn_up.weight` [896, 4864] (F16)
    - **Size**: 896×4864 = 4,358,144 F16 parameters (~8.7M bytes)
    - **Kernel Used**: `MatmulDispatchOp` (MLX for large N=4864)
  - **SwiGLU Activation**: gate_silu(gate) * up
    - **Kernel Used**: `SwiGLUOp` (custom activation kernel)
    - **CPU/GPU Work**: GPU operation
  - **Down Projection**: [batch*seq, 4864] × [4864, 896] → [batch*seq, 896]
    - **Weight**: `blk.{i}.ffn_down.weight` [4864, 896] (F16)
    - **Size**: 4864×896 = 4,358,144 F16 parameters (~8.7M bytes)
    - **Kernel Used**: `MatmulDispatchOp` (MLX for large M=4864)
  - **CPU/GPU Work**: Three GPU matmuls + activation, all remain on GPU
  - **Sync Points**: None

### 3. Final Processing
- **Residual Connection**: Add input to attention and MLP outputs (for each layer)
  - **Kernel Used**: Element-wise add operations
  - **CPU/GPU Work**: GPU operation
  - **Sync Points**: None

- **Final RMSNorm**: Normalize after all transformer layers
  - **Input**: [batch, seq, 896] (post-MLP result of last layer)
  - **Tensor**: `output_norm.weight` [896] (F32)
  - **Kernel Used**: `RMSNormOp`
  - **CPU/GPU Work**: GPU operation
  - **Sync Points**: None

- **Output Projection**: Final matmul to convert to vocab_size logits
  - **Input**: [batch*seq, 896]
  - **Weight**: `output.weight` [151936, 896] (F16) (likely same as token_embd.weight but transposed)
    - **Size**: 151936×896 = 136,134,656 F16 parameters (~272M bytes)
  - **Operation**: [batch*seq, 896] × [896, 151936] → [batch*seq, 151936]
  - **Kernel Used**: `MatmulDispatchOp` (large N=151936 → likely MLX)
  - **CPU/GPU Work**: GPU operation
  - **Sync Points**: **YES** - GPU to CPU sync required to read logits for sampling

### 4. Sampling
- **Logits Processing**: Download logits from GPU, apply temperature/top-p/top-k sampling
  - **Input**: [batch, seq, 151936] logits tensor from GPU
  - **CPU/GPU Work**: CPU operation (sampling functions in `gpu_sample_top_k_top_p`)
  - **Sync Points**: **YES** - GPU sync to download logits to CPU memory
  - **Operation**: Download tensor from GPU, apply sampling strategy to select next token ID

### 5. KV Cache Management
- **For autoregressive generation**, KV caches are maintained across tokens
- **New tokens' K/V values** are appended to the cache (GPU operations)
- **Existing cached K/V values** are retrieved and repeated for attention computation (GPU operations)
- **Cache Size Calculation**: Each layer requires 2×[batch, n_heads, max_seq, head_dim] F16 storage
  - Per layer: 2 × 1 × 14 × 32768 × 64 × 2 bytes = ~112MB per layer
  - Total: 24 × ~112MB = ~2.7GB for full KV cache

### Performance Bottlenecks Summary:
1. **Large matmuls**: Output projection (896×151936) and FFN operations (896×4864, 4864×896) dominate computation
2. **KV cache management**: Storage and retrieval of large KV tensors affects memory bandwidth
3. **GPU-CPU sync points**: Only during logits download for sampling - minimize these
4. **Matmul dispatcher effectiveness**: Depends on selecting optimal kernel for each shape (MLX vs MPS vs custom GEMV)

The matmul dispatcher and softmax dispatcher are the key optimization points that select from multiple backends (MLX, MPS, custom GEMV kernels, vec/block softmax) based on tensor dimensions and device capabilities.

---

## Current Sync Points Analysis

### 1. Mandatory Sync Points (per token):
- **Logits Download**: GPU → CPU sync to read logits for sampling (main bottleneck)

### 2. Indirect Sync Points (per token):
- **Token ID Processing**: CPU → GPU sync for token embedding lookup
- **Token Decoding**: CPU operation to convert token ID to text

### 3. KV Cache Operations:
- All happen on GPU with no explicit sync points (efficient)

---

## Proposed Architectural Improvements to Reduce Sync Points

### 1. GPU-Based Sampling (Eliminates Logits Download Sync)
- **Current**: Download logits to CPU → sample → return token ID to GPU
- **Improved**: Implement sampling kernels directly in Metal
  - Top-k selection kernel: Find top-k indices directly on GPU
  - Softmax + sampling kernel: Apply temperature, compute probabilities, sample on GPU
  - Output single token ID back to CPU (much smaller data transfer)
- **Benefit**: Eliminates the main sync point for every generated token
- **Priority**: **HIGH** - This would have the most dramatic impact on performance

### 2. GPU-Based Token Embedding (Eliminates CPU→GPU Transfer)
- **Current**: CPU passes token ID → GPU embedding lookup
- **Improved**: Keep a small circular buffer of recent token IDs on GPU and run embedding lookup on GPU
- **Benefit**: No need to pass token IDs from CPU to GPU each iteration
- **Priority**: **MEDIUM** - Reduces a recurring CPU-GPU transfer

### 3. Streaming/Async Processing Architecture
- **Implementation**: Pipeline the operations so GPU is never idle
  - While CPU processes current token, GPU can start processing next step
  - Use multiple command buffers in flight
- **Benefit**: Overlaps CPU and GPU work, hiding sync latencies
- **Priority**: **MEDIUM** - Improves overall throughput

### 4. GPU-Based KV Cache Management
- **Current**: Some KV cache operations may involve CPU coordination
- **Improved**: Implement the entire KV caching and retrieval process in Metal
  - Use Metal compute kernels for cache indexing and retrieval
  - Implement a GPU-managed circular buffer for KV cache
- **Benefit**: Eliminates CPU involvement in cache management
- **Priority**: **LOW** - Current implementation is already GPU-heavy

### 5. Batched Generation for Multiple Sequences
- **Implementation**: Instead of single-token autoregressive generation, process multiple sequences or tokens in parallel
- **Benefit**: Amortizes sync costs across multiple tokens/sequences
- **Priority**: **MEDIUM** - Good for throughput but may increase latency

### 6. GPU-Resident Tokenizer/Decoder
- **Current**: Token decoding happens on CPU in `tokenizer.decode_token_arc()`
- **Improved**: Implement a basic GPU-based tokenizer that can handle common tokens
- **Benefit**: Keep the entire generation pipeline on GPU
- **Priority**: **LOW** - Decoding is relatively fast, but would improve pipeline completeness

### 7. Pipeline Stages with GPU Command Buffers
- **Implementation**: Structure the generation as a pipeline where each token's processing happens in a different stage
- **Benefit**: While current token is being sampled on CPU, next token's attention computation can already be queued on GPU
- **Priority**: **HIGH** - Would significantly improve pipeline efficiency

---

## Priority Recommendations

### Immediate High-Impact Changes:
1. **GPU-based sampling** - Would eliminate the primary per-token sync bottleneck
2. **Pipeline architecture** - Would allow overlapping of CPU sampling and GPU computation

### Medium-Term Improvements:
1. **GPU-based token embedding** - Reduces another per-token transfer
2. **Batched processing** - Improves overall throughput

### Long-Term Enhancements:
1. **GPU tokenizer/decoder** - Completes the GPU-only pipeline
2. **Advanced memory management** - Optimizes KV cache usage