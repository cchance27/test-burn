#include <metal_stdlib>
using namespace metal;

kernel void copy_kernel(device const float *in [[buffer(0)]],
                        device float *out [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = in[id];
}
