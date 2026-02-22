#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// Cooperative tile load for a [TileN=32, D=64] block.
// - `src` must already point at the first element of the tile (row 0, col 0).
// - `row_stride` is in elements (half), not bytes.
// - `limit` is the number of valid rows in this tile (<= 32); remaining rows are zero-filled.
// - `tid` is the linear thread index in the threadgroup [0, 255].
inline void load_tile(
    const device half* src,
    threadgroup half* dst,
    uint row_stride,
    uint limit,
    uint tid
) {
    // Each thread loads 8 consecutive half elements (16 bytes).
    // 256 threads * 8 half = 2048 half = 32 * 64.
    uint row_in_tile = tid / 8;
    uint col_in_tile = (tid % 8) * 8;
    
    if (row_in_tile < 32) {
        bool active = row_in_tile < limit;
        
        const device ulong* src_u = (const device ulong*)(src + row_in_tile * row_stride + col_in_tile);
        threadgroup ulong* dst_u = (threadgroup ulong*)(dst + row_in_tile * 64 + col_in_tile);
        if (active) {
            // Copy 16 bytes via 2x 64-bit loads/stores.
            // `col_in_tile` is a multiple of 8 half elements, so this is 16-byte aligned.
            dst_u[0] = src_u[0];
            dst_u[1] = src_u[1];
        } else {
            dst_u[0] = 0;
            dst_u[1] = 0;
        }
    }
}

// Cooperative tile load for a [TileN=32, D=128] block.
// 256 threads * 16 half = 4096 half = 32 * 128.
inline void load_tile_d128(
    const device half* src,
    threadgroup half* dst,
    uint row_stride,
    uint limit,
    uint tid
) {
    uint row_in_tile = tid / 8;
    uint col_in_tile = (tid % 8) * 16;

    if (row_in_tile < 32) {
        bool active = row_in_tile < limit;

        const device ulong* src_u = (const device ulong*)(src + row_in_tile * row_stride + col_in_tile);
        threadgroup ulong* dst_u = (threadgroup ulong*)(dst + row_in_tile * 128 + col_in_tile);
        if (active) {
            // Copy 32 bytes via 4x 64-bit loads/stores.
            dst_u[0] = src_u[0];
            dst_u[1] = src_u[1];
            dst_u[2] = src_u[2];
            dst_u[3] = src_u[3];
        } else {
            dst_u[0] = 0;
            dst_u[1] = 0;
            dst_u[2] = 0;
            dst_u[3] = 0;
        }
    }
}
