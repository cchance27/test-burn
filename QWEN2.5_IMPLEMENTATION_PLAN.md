# Qwen2.5 Implementation Plan for Metallic Engine

This document outlines the steps required to fully support Qwen2.5 inference using our GGUF implementation and Metallic engine.

## Current Status

We have successfully implemented:
1. GGUF file parsing and loading with memory mapping
2. Support for F32, F64, and Q8 quantized tensors
3. Integration with Metallic tensors
4. SIMD-optimized dequantization for Apple Silicon
5. Basic memory management (offloading/blockswapping)
6. Core operations: SDPA, LayerNorm, GELU, MatMul, Softmax

## Missing Components for Full Qwen2.5 Support

### 1. Tokenizer Implementation

**Current Status**: No tokenizer implemented
**Requirements**: 
- BPE tokenizer with vocabulary from GGUF metadata
- Support for special tokens (BOS, EOS, padding)
- Text to token conversion and vice versa
- Support for Qwen2.5's specific tokenizer (GPT-2 style with merges)

**Implementation Plan**:
- Create a new `tokenizer` module in `src/`
- Implement BPE tokenizer with vocabulary from GGUF metadata
- Handle special tokens based on GGUF metadata
- Implement encoding/decoding functions

### 2. Model Architecture Implementation

**Current Status**: Basic operations available, but no model structure
**Requirements**:
- Qwen2.5 transformer architecture implementation
- 24 transformer blocks as per metadata
- Embedding layer (token_embd.weight)
- Output normalization layer (output_norm.weight)
- Each transformer block contains:
  - Attention norm (attn_norm.weight)
  - Attention mechanism (attn_q, attn_k, attn_v, attn_output)
  - Feed-forward network (ffn_gate, ffn_up, ffn_down)
  - FFN norm (ffn_norm.weight)

**Implementation Plan**:
- Create Qwen2.5 specific model implementation
- Implement transformer block structure
- Handle rotary positional embeddings (RoPE)
- Implement RMSNorm instead of LayerNorm for attention and FFN
- Handle grouped-query attention (14 heads for Q, 2 heads for K/V)

### 3. Additional Operations Needed

**Current Status**: Core operations implemented
**Missing Operations**:
- RMSNorm (different from LayerNorm)
- Rotary Positional Embeddings (RoPE)
- SiLU activation for FFN gate
- Proper tensor indexing and reshaping for attention mechanisms

**Implementation Plan**:
- Implement RMSNorm operation in Metallic
- Implement RoPE operation
- Add SiLU activation (similar to GELU)
- Enhance tensor operations for proper attention mechanism implementation

### 4. Inference Pipeline

**Current Status**: No complete inference pipeline
**Requirements**:
- Complete end-to-end inference pipeline
- Tokenization → Model inference → Detokenization
- Support for different generation parameters (temperature, top-p, etc.)
- Support for chat templates

**Implementation Plan**:
- Create inference pipeline that connects all components
- Implement generation loop with sampling
- Support for chat templates from GGUF metadata
- Add generation parameters support

### 5. Memory Management Enhancements

**Current Status**: Basic offloading/blockswapping implemented
**Requirements**:
- More sophisticated memory management for large models
- Layer-by-layer loading/unloading
- GPU memory optimization
- CPU offloading for layers that don't fit in GPU memory

**Implementation Plan**:
- Enhance memory management system
- Implement layer-wise memory optimization
- Add automatic offloading based on available GPU memory

## Detailed Implementation Steps

### Phase 1: Tokenizer Implementation (1-2 days)
1. Create `src/tokenizer` module
2. Implement BPE tokenizer with GGUF metadata integration
3. Handle special tokens (BOS: 151643, EOS: 151645, padding: 151665)
4. Implement encoding/decoding functions
5. Add tests for tokenizer functionality

### Phase 2: Additional Operations (2-3 days)
1. Implement RMSNorm operation (different from LayerNorm)
2. Implement RoPE (Rotary Positional Embeddings)
3. Implement SiLU activation function
4. Enhance tensor operations for attention mechanisms
5. Add tests for all new operations

### Phase 3: Model Architecture (3-4 days)
1. Create Qwen2.5 model structure
2. Implement transformer blocks with all components
3. Handle grouped-query attention (14 Q heads, 2 K/V heads)
4. Implement embedding and output layers
5. Connect all components into a working model
6. Add tests for model components

### Phase 4: Inference Pipeline (2-3 days)
1. Create end-to-end inference pipeline
2. Implement generation loop with sampling
3. Add support for chat templates
4. Implement generation parameters (temperature, top-p, etc.)
5. Add performance optimizations
6. Add tests for inference pipeline

### Phase 5: Memory Management Enhancements (2-3 days)
1. Enhance memory management system
2. Implement layer-wise loading/unloading
3. Add automatic offloading based on GPU memory
4. Optimize memory usage for inference
5. Add tests for memory management

## Technical Details from Metadata

### Model Configuration:
- Architecture: Qwen2
- Block count: 24 transformer blocks
- Context length: 32,768 tokens
- Embedding length: 896 dimensions
- Feed-forward length: 4,864 dimensions
- Attention heads: 14 (Q), 2 (K/V)
- RoPE frequency base: 1,000,000
- Layer norm RMS epsilon: 1e-6

### Tokenizer:
- Model type: GPT-2
- Preprocessing: qwen2
- Vocabulary size: 151,936 tokens
- Special tokens:
  - BOS token ID: 151643
  - EOS token ID: 151645
  - Padding token ID: 151665

## Performance Considerations

1. **Quantization**: Our Q8_0/Q8_1 support with SIMD optimization should provide good performance
2. **Memory Management**: Need to implement efficient layer loading/unloading for large context
3. **Attention Optimization**: Grouped-query attention should be more efficient than full attention
4. **RoPE Implementation**: Should be optimized for performance
5. **Batch Processing**: Consider support for batched inference

## Testing Strategy

1. Unit tests for each new component
2. Integration tests for the complete pipeline
3. Performance benchmarks against reference implementations
4. Accuracy validation with known inputs/outputs
5. Memory usage profiling

## Dependencies

1. Existing GGUF implementation
2. Existing Metallic operations (SDPA, LayerNorm, GELU, MatMul, Softmax)
3. New tokenizer implementation
4. New operations (RMSNorm, RoPE, SiLU)
5. Enhanced memory management

## Timeline

Total estimated time: 10-15 days for complete implementation

This plan provides a roadmap for implementing full Qwen2.5 support in our Metallic engine with GGUF format.