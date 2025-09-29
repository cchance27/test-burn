using namespace metal;

// Elementwise divide: out[i] = a[i] / b[i]
kernel void div_kernel(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       constant uint& total_elements [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= total_elements) return;
    out[gid] = a[gid] / b[gid];
}
    