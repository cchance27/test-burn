#ifndef METALLIC_TILE_LOADER_METAL
#define METALLIC_TILE_LOADER_METAL

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// TileLoader - Primary primitive for cooperative tile loading
// =============================================================================
//
// Cooperatively loads a BROWS x BCOLS tile from device memory to threadgroup memory.
// Uses Policies (PolicyF16, PolicyQ8, etc.) to handle dequantization.
//
// Refactored to use base pointer + global element offset for robustness.
//
template<
    typename Policy,     // Dequantization policy
    typename T,          // Output type in threadgroup (usually half)
    short BROWS,         // Tile height
    short BCOLS,         // Tile width
    short dst_ld,        // Stride in threadgroup memory
    bool transpose,      // Whether to transpose during load (not used by TileLoader itself but for stride calc)
    short tgp_size       // Total threads in threadgroup
>
struct TileLoader {
    // Each thread loads n_reads elements
    static constant short total_elements = BROWS * BCOLS;
    static constant short n_reads = total_elements / tgp_size;
    static constant short TCOLS = BCOLS / n_reads;
    static constant short TROWS = tgp_size / TCOLS;
    
    const int src_ld;
    const int k_stride_elements;
    
    const device uchar* src_base;
    const device uchar* scales;
    const uint weights_per_block;
    const uint row_idx_offset; // Starting row/col for scale indexing
    const uint blocks_per_k;
    
    uint k_offset; // Global offset along the reduction dimension
    const short bi, bj;
    
    threadgroup T* dst;
    
    /// Constructor
    METAL_FUNC TileLoader(
        const device uchar* src_,
        const int src_ld_,
        threadgroup T* dst_,
        const device uchar* scales_,
        const uint weights_per_block_,
        const uint row_idx_offset_,
        const uint blocks_per_k_,
        const uint N_, // Unused for now but kept for compatibility
        ushort simd_group_id,
        ushort simd_lane_id
    ) : src_ld(src_ld_),
        k_stride_elements(transpose ? BCOLS : BROWS), 
        src_base(src_),
        scales(scales_),
        weights_per_block(weights_per_block_),
        row_idx_offset(row_idx_offset_),
        blocks_per_k(blocks_per_k_),
        k_offset(0),
        bi((simd_group_id * 32 + simd_lane_id) / TCOLS),
        bj(n_reads * ((simd_group_id * 32 + simd_lane_id) % TCOLS)),
        dst(dst_ + bi * dst_ld + bj)
    {
        // dst points to thread's starting element in TGP memory
    }
    
    /// Load tile into threadgroup memory
    METAL_FUNC void load_unsafe() const {
        #pragma unroll
        for (short i = 0; i < BROWS; i += TROWS) {
            if (bi + i < BROWS) {
                #pragma unroll
                for (short j = 0; j < n_reads; j++) {
                    // Determine logical K and N indices
                    ulong K_idx, N_idx;
                    if (transpose) {
                        // B in memory is [N, K]. BROWS=BN, BCOLS=BK. 
                        // bi+i is N-index, bj+j is K-index.
                        N_idx = row_idx_offset + bi + i;
                        K_idx = k_offset + bj + j;
                    } else {
                        // B in memory is [K, N]. BROWS=BK, BCOLS=BN. 
                        // bi+i is K-index, bj+j is N-index.
                        K_idx = k_offset + bi + i;
                        N_idx = row_idx_offset + bj + j;
                    }
                    
                    // Physical row/col in memory
                    ulong global_row = transpose ? N_idx : K_idx;
                    ulong global_col = transpose ? K_idx : N_idx;
                    ulong offset = global_row * (ulong)src_ld + global_col;
                    
                    // Load weights via policy
                    float w_val[1];
                    Policy::template load_weights<1>(src_base, offset, w_val);
                    
                    // Load scale indexing
                    ulong block_idx = N_idx * blocks_per_k + (K_idx / weights_per_block);
                    half scale = Policy::load_scale(scales, block_idx);
                    
                    dst[i * dst_ld + j] = (T)(w_val[0] * (float)scale);
                }
            }
        }
    }
    
    /// Load tile with bounds checking
    METAL_FUNC void load_safe(short2 src_tile_dim) const {
        // src_tile_dim defines the valid (K, N) or (N, K) region for this tile
        #pragma unroll
        for (short i = 0; i < BROWS; i += TROWS) {
            #pragma unroll
            for (short j = 0; j < n_reads; j++) {
                bool valid = (bi + i < src_tile_dim.y) && (bj + j < src_tile_dim.x);
                if (valid) {
                    ulong K_idx, N_idx;
                    if (transpose) {
                        N_idx = row_idx_offset + bi + i;
                        K_idx = k_offset + bj + j;
                    } else {
                        K_idx = k_offset + bi + i;
                        N_idx = row_idx_offset + bj + j;
                    }
                    
                    ulong global_row = transpose ? N_idx : K_idx;
                    ulong global_col = transpose ? K_idx : N_idx;
                    ulong offset = (ulong)global_row * src_ld + global_col;
                    
                    float w_val[1];
                    Policy::template load_weights<1>(src_base, offset, w_val);
                    
                    ulong block_idx = N_idx * blocks_per_k + (K_idx / weights_per_block);
                    half scale = Policy::load_scale(scales, block_idx);
                    
                    dst[i * dst_ld + j] = (T)(w_val[0] * (float)scale);
                } else {
                    dst[i * dst_ld + j] = T(0);
                }
            }
        }
    }
    
    /// Advance to next K block
    METAL_FUNC void next() {
        k_offset += k_stride_elements;
    }
};

// =============================================================================
// SimpleTileLoader - Optimized for F16 activations (no dequant)
// =============================================================================
template<
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    bool transpose, // Not used for loading but for consistency
    short tgp_size
>
struct SimpleTileLoader {
    static constant short total_elements = BROWS * BCOLS;
    static constant short n_reads = total_elements / tgp_size;
    static constant short TCOLS = BCOLS / n_reads;
    static constant short TROWS = tgp_size / TCOLS;
    
    const int src_ld;
    const int tile_stride;
    
    const device T* src;
    threadgroup T* dst;
    
    const short bi, bj;
    
    METAL_FUNC SimpleTileLoader(
        const device T* src_,
        const int src_ld_,
        threadgroup T* dst_,
        ushort simd_group_id,
        ushort simd_lane_id
    ) : src_ld(src_ld_),
        tile_stride(transpose ? BCOLS : BROWS * src_ld_),
        src(src_), 
        dst(dst_),
        bi((simd_group_id * 32 + simd_lane_id) / TCOLS),
        bj(n_reads * ((simd_group_id * 32 + simd_lane_id) % TCOLS))
    {
        // Shift pointers to thread-local start
        src += bi * src_ld + bj;
        dst += bi * dst_ld + bj;
    }
    
    METAL_FUNC void load_unsafe() const {
        #pragma unroll
        for (short i = 0; i < BROWS; i += TROWS) {
            if (bi + i < BROWS) {
                #pragma unroll
                for (short j = 0; j < n_reads; j++) {
                    dst[i * dst_ld + j] = src[i * src_ld + j];
                }
            }
        }
    }
    
    METAL_FUNC void load_safe(short2 src_tile_dim) const {
        src_tile_dim = src_tile_dim - short2(bj, bi);
        #pragma unroll
        for (short i = 0; i < BROWS; i += TROWS) {
            #pragma unroll
            for (short j = 0; j < n_reads; j++) {
                bool valid = (i < src_tile_dim.y) && (j < src_tile_dim.x);
                dst[i * dst_ld + j] = valid ? src[i * src_ld + j] : T(0);
            }
        }
    }
    
    METAL_FUNC void next() {
        src += tile_stride;
    }
};

#endif // METALLIC_TILE_LOADER_METAL
