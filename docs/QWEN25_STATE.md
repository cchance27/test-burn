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
- **Logits Processing**: Fused top-k/top-p sampling is performed directly on the GPU.
  - **Input**: [batch, seq, 151936] logits tensor on GPU
  - **CPU/GPU Work**: GPU operation (`SampleTopKFused` Metal kernel).
  - **Sync Points**: **Minimal**. A small sync is required to read the single resulting token ID (a `u32`) from the GPU. The large logits tensor is never downloaded.
  - **Operation**: A custom Metal kernel performs temperature scaling, top-k selection, top-p nucleus sampling, and random sampling in a single, fused operation.

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
3. **Matmul dispatcher effectiveness**: Depends on selecting optimal kernel for each shape (MLX vs MPS vs custom GEMV)

The matmul dispatcher and softmax dispatcher are the key optimization points that select from multiple backends (MLX, MPS, custom GEMV kernels, vec/block softmax) based on tensor dimensions and device capabilities.

---

## Current Sync Points Analysis

### 1. Mandatory Sync Points (per token):
- **Token ID Readback**: A GPU → CPU sync to read the single selected token ID. This is a very small data transfer (4 bytes) and is a massive improvement over downloading the entire logits tensor.

### 2. Indirect Sync Points (per token):
- **Token ID Processing**: CPU → GPU sync for token embedding lookup
- **Token Decoding**: CPU operation to convert token ID to text

### 3. KV Cache Operations:
- All happen on GPU with no explicit sync points (efficient)

---

## Architectural Improvements

### 1. GPU-Based Sampling (Eliminates Logits Download Sync) - ✅ DONE
- **Status**: **Implemented**. The `SampleTopKFused` Metal kernel performs the entire sampling process on the GPU.
- **Benefit**: The main sync point for every generated token has been eliminated, resulting in a significant performance improvement.

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
- **Priority**: **HIGH** - This is the next major performance improvement to tackle.

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

---

## Latest Profiling Snapshot (63 tok/s, 16ms loop latency)

- **Throughput**: 63 tokens/sec (target ≈150 tok/s)
- **Generation loop latency**: ~16ms per token
- **Profiling (profilingdisabled.jsonl)**:
  - `ResourceCacheAccess`: 14,287 events – indicates heavy use of cache lookups per token
  - `InternalKernelCompleted:generation_loop`: 3,236 ops totaling 609.5ms (avg 0.2ms, max 19.4ms)
  - `GpuOpCompleted`: 1,220 ops totaling 3.0ms (avg 0.002ms)
  - Minimal GPU compute time vs high kernel scheduling overhead implies that command dispatch and CPU-side orchestration dominate token time, not raw math throughput.

### Immediate Focus Areas

1. **Command Buffer Pipelining**
   - Current loop appears mostly serialized; GPU dispatch time is dwarfed by CPU coordination.
   - Maintain 2–3 command buffers in flight, pre-populating the next token's kernels while the current buffer executes to hide submission overhead.
   - Batch Metal command encoder creation (reuse encoder objects, avoid per-token setup costs) and issue fences only when token ID must be read back.

2. **Resource Cache Hot Path Audit**
   - `ResourceCacheAccess` dominance suggests frequent lookups for recurrent tensors (e.g., matmul weights, rope caches).
   - Introduce persistent handles for the static resources needed every token to avoid hash-map probing; prebind buffers within a `generation_plan` struct during session initialization.
   - Where the cache is unavoidable, profile the hash/equality path and consider fxhash-like alternatives to reduce lookup time.

3. **GPU-Resident Token Staging**
   - Implement the circular token ID buffer outlined above so the embedding kernel can run entirely on GPU, removing a CPU→GPU bounce per step.
   - Pair with a lightweight GPU-side argmax-to-ID staging kernel that writes to a mapped buffer, so the CPU only reads when strictly necessary.

4. **Microkernel Fusion Opportunities**
   - The `generation_loop` max latency spikes (≈19ms) hint at occasional cache misses or unfused sequences.
   - Investigate fusing RMSNorm + matmul epilogues (where mathematically valid) to reduce command count.
   - Evaluate whether Q/K/V projections can be dispatched as a single GEMM with grouped bias to reduce three separate command submissions.

### Medium-Term Investigations

1. **Async Tokenizer Path**
   - Offload decode work to a background CPU thread so the main thread can immediately queue the next GPU workload after issuing readback.

2. **Command Submission Telemetry**
   - Add per-token logging of command-buffer creation, commit, and completion times to quantify the CPU orchestration cost and validate that pipelining improvements move the bottleneck back to GPU math.