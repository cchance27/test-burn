#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// Keep tile/vector aliases centralized so storage widening is localized.
#if METALLIC_FASTPATH_INPUT_HALF
typedef half FlashTileT;
typedef half2 FlashVec2T;
typedef half4 FlashVec4T;
#define SDPA_PREFILL_DOUBLE_BUFFER 1
#else
typedef float FlashTileT;
typedef float2 FlashVec2T;
typedef float4 FlashVec4T;
#define SDPA_PREFILL_DOUBLE_BUFFER 0
#endif

// D128 uses [32 x 128] tiles. D64 paths use the lower half.
#define SDPA_PREFILL_TILE_BANK_STRIDE (32 * 128)

#if SDPA_PREFILL_DOUBLE_BUFFER
#define SDPA_PREFILL_DECLARE_SHARED(NAME_K, NAME_V) \
    threadgroup FlashTileT NAME_K[2 * SDPA_PREFILL_TILE_BANK_STRIDE]; \
    threadgroup FlashTileT NAME_V[2 * SDPA_PREFILL_TILE_BANK_STRIDE]
#else
#define SDPA_PREFILL_DECLARE_SHARED(NAME_K, NAME_V) \
    threadgroup FlashTileT NAME_K[SDPA_PREFILL_TILE_BANK_STRIDE]; \
    threadgroup FlashTileT NAME_V[SDPA_PREFILL_TILE_BANK_STRIDE]
#endif

#define SDPA_PREFILL_ENGINE_FA1 0u
#define SDPA_PREFILL_ENGINE_FA2 1u

// Cooperative tile load for a [TileN=32, D=64] block.
// - `src` must already point at the first element of the tile (row 0, col 0).
// - `row_stride` is in elements, not bytes.
// - `limit` is the number of valid rows in this tile (<= 32); remaining rows are zero-filled.
// - `tid` is the linear thread index in the threadgroup [0, 255].
inline void load_tile(
    const device InputStorageT* src,
    threadgroup FlashTileT* dst,
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
        
#if METALLIC_FASTPATH_INPUT_HALF
        const device ulong* src_u = (const device ulong*)((const device FlashTileT*)src + row_in_tile * row_stride + col_in_tile);
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
#else
        threadgroup FlashTileT* dst_row = dst + row_in_tile * 64 + col_in_tile;
        if (active) {
            const device InputStorageT* src_row = src + row_in_tile * row_stride + col_in_tile;
            #pragma unroll
            for (uint i = 0; i < 8; ++i) {
                dst_row[i] = (FlashTileT)src_row[i];
            }
        } else {
            #pragma unroll
            for (uint i = 0; i < 8; ++i) {
                dst_row[i] = (FlashTileT)0.0f;
            }
        }
#endif
    }
}

// Cooperative tile load for a [TileN=32, D=128] block.
// 256 threads * 16 half = 4096 half = 32 * 128.
inline void load_tile_d128(
    const device InputStorageT* src,
    threadgroup FlashTileT* dst,
    uint row_stride,
    uint limit,
    uint tid
) {
    uint row_in_tile = tid / 8;
    uint col_in_tile = (tid % 8) * 16;

    if (row_in_tile < 32) {
        bool active = row_in_tile < limit;

#if METALLIC_FASTPATH_INPUT_HALF
        const device ulong* src_u = (const device ulong*)((const device FlashTileT*)src + row_in_tile * row_stride + col_in_tile);
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
#else
        threadgroup FlashTileT* dst_row = dst + row_in_tile * 128 + col_in_tile;
        if (active) {
            const device InputStorageT* src_row = src + row_in_tile * row_stride + col_in_tile;
            #pragma unroll
            for (uint i = 0; i < 16; ++i) {
                dst_row[i] = (FlashTileT)src_row[i];
            }
        } else {
            #pragma unroll
            for (uint i = 0; i < 16; ++i) {
                dst_row[i] = (FlashTileT)0.0f;
            }
        }
#endif
    }
}
