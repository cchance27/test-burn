#include <metal_stdlib>
using namespace metal;

// Define per-dtype variants similar to other kernels
#define FOR_EACH_FLOAT_TYPE(OP) \
    OP(float, float, f32) \
    OP(half, float, f16)

// Each thread writes one output element. Grid size = batch * seq * d_model.
// Inputs:
//  - table: [vocab_size, d_model] row-major
//  - indices: [batch * seq] of uint token ids
//  - out: [batch, seq, d_model] flattened
//  - d_model: feature dimension
//  - total_elements: out length
//  - vocab_size: bound-check for indices
#define DEFINE_EMBEDDING_LOOKUP_KERNEL(SCALAR, ACCUM, SUFFIX) \
kernel void embedding_lookup_kernel_##SUFFIX(device const SCALAR* table [[buffer(0)]], \
                                             device const uint* indices [[buffer(1)]], \
                                             device SCALAR* out [[buffer(2)]], \
                                             constant uint& d_model [[buffer(3)]], \
                                             constant uint& total_elements [[buffer(4)]], \
                                             constant uint& vocab_size [[buffer(5)]], \
                                             uint gid [[thread_position_in_grid]]) { \
    if (gid >= total_elements) return; \
    uint dm = d_model; \
    uint pos = gid / dm; \
    uint feat = gid - pos * dm; \
    uint tok = indices[pos]; \
    if (tok >= vocab_size) { \
        out[gid] = (SCALAR)0; \
        return; \
    } \
    uint src = tok * dm + feat; \
    out[gid] = table[src]; \
}

FOR_EACH_FLOAT_TYPE(DEFINE_EMBEDDING_LOOKUP_KERNEL)

#undef DEFINE_EMBEDDING_LOOKUP_KERNEL
#undef FOR_EACH_FLOAT_TYPE

