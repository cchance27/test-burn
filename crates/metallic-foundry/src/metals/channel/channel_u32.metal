#include <metal_stdlib>
using namespace metal;

// Buffers:
// - header: u32[4] { write_idx, read_idx, capacity, flags }
// - data:   u32[capacity]
// - value:  u32[1] (single scalar value to push)

kernel void channel_u32_init(
    device uint* header [[buffer(0)]],
    device uint* data [[buffer(1)]],
    constant ChannelU32Params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) return;

    const uint capacity = params.capacity;

    header[0] = 0u;         // write_idx
    header[1] = 0u;         // read_idx
    header[2] = capacity;   // capacity
    header[3] = 0u;         // flags

    // v1: clear data for deterministic tests/debugging.
    for (uint i = 0u; i < capacity; ++i) {
        data[i] = 0u;
    }
}

kernel void channel_u32_push(
    device uint* header [[buffer(0)]],
    device uint* data [[buffer(1)]],
    const device uint* value [[buffer(2)]],
    constant ChannelU32Params& _params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) return;

    const uint capacity = header[2];
    if (capacity == 0u) return;

    const uint write_idx = header[0];
    const uint slot = write_idx % capacity;
    data[slot] = value[0];
    header[0] = write_idx + 1u;
}

kernel void channel_u32_push_scalar(
    device uint* header [[buffer(0)]],
    device uint* data [[buffer(1)]],
    constant uint& value [[buffer(2)]],
    constant ChannelU32Params& _params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) return;

    const uint capacity = header[2];
    if (capacity == 0u) return;

    const uint write_idx = header[0];
    const uint slot = write_idx % capacity;
    data[slot] = value;
    header[0] = write_idx + 1u;
}
