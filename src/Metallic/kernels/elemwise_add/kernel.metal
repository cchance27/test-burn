using namespace metal;

// Elementwise add: out[i] = a[i] + b[i]
kernel void add_kernel(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       constant uint& total_elements [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= total_elements) return;
    out[gid] = a[gid] + b[gid];
}

// Broadcast add for bias: out[i] = a[i] + b[i % b_len], where b_len is the broadcast dimension (e.g., bias len)
kernel void broadcast_add_kernel(device const float* a [[buffer(0)]],
                                 device const float* b [[buffer(1)]],
                                 device float* out [[buffer(2)]],
                                 constant uint& total_elements [[buffer(3)]],
                                 constant uint& b_len [[buffer(4)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= total_elements) return;
    uint b_idx = gid % b_len;
    out[gid] = a[gid] + b[b_idx];
}
    