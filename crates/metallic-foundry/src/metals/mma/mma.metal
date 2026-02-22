#ifndef METALLIC_MMA_METAL
#define METALLIC_MMA_METAL

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

// =============================================================================
// SimdgroupMma - Hardware-accelerated matrix multiply-accumulate
// =============================================================================
//
// Follows the MLX GEMM pattern for reliable simdgroup_matrix usage.
// Uses thread_elements() for loading/storing with explicit type casting.
// This is POLICY-AGNOSTIC - operates on already-dequantized threadgroup memory.
//
// Usage:
//   SimdgroupMma<half, half, 32, 32, 16, 2, 2, false, false, 20, 36> mma(sgid, slid);
//   for (int k = 0; k < K_iters; k++) {
//       mma.mma(As, Bs);
//   }
//   mma.store_result(D, ldd);
//

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

/// Simdgroup matrix multiply-accumulate for tiled GEMM.
/// Follows MLX's BlockMMA pattern exactly.
template<
    typename T,          // Element type in threadgroup memory (half)
    typename U,          // Output element type (half)
    int BM,              // Block tile M dimension
    int BN,              // Block tile N dimension
    int BK,              // Block tile K dimension
    int WM,              // Warp tiles in M
    int WN,              // Warp tiles in N
    bool transpose_a,    // Whether A is transposed
    bool transpose_b,    // Whether B is transposed
    short lda_tgp,       // Leading dimension of A in threadgroup memory
    short ldb_tgp,       // Leading dimension of B in threadgroup memory
    typename AccumType = float
>
struct SimdgroupMma {
    // Warp tile strides
    STEEL_CONST short TM_stride = 8 * WM;
    STEEL_CONST short TN_stride = 8 * WN;
    
    // Number of simdgroup tiles per warp
    STEEL_CONST short TM = BM / TM_stride;
    STEEL_CONST short TN = BN / TN_stride;
    
    // Strides for simdgroup load within threadgroup tile
    STEEL_CONST short simd_stride_a = transpose_a ? TM_stride : TM_stride * lda_tgp;
    STEEL_CONST short simd_stride_b = transpose_b ? TN_stride * ldb_tgp : TN_stride;
    
    // Jump between adjacent elements
    STEEL_CONST short jump_a = transpose_a ? lda_tgp : 1;
    STEEL_CONST short jump_b = transpose_b ? ldb_tgp : 1;
    
    // Stride for iterating over K within BK
    STEEL_CONST short tile_stride_a = transpose_a ? 8 * lda_tgp : 8;
    STEEL_CONST short tile_stride_b = transpose_b ? 8 : 8 * ldb_tgp;
    
    // Simdgroup matrix registers
    simdgroup_matrix<AccumType, 8, 8> Asimd[TM];
    simdgroup_matrix<AccumType, 8, 8> Bsimd[TN];
    simdgroup_matrix<AccumType, 8, 8> results[TM * TN];
    
    // Thread position within simdgroup
    const short tm;  // Warp row offset
    const short tn;  // Warp col offset
    short sm;        // Thread row within 8x8 tile
    short sn;        // Thread col within 8x8 tile
    short As_offset; // Starting offset in As
    short Bs_offset; // Starting offset in Bs
    
    /// Constructor - computes thread position using MLX's formula
    METAL_FUNC SimdgroupMma(
        ushort simd_group_id [[simdgroup_index_in_threadgroup]],
        ushort simd_lane_id [[thread_index_in_simdgroup]]
    ) : tm(8 * (simd_group_id / WN)),
        tn(8 * (simd_group_id % WN))
    {
        // Compute thread position within 8x8 simdgroup matrix (MLX formula)
        short qid = simd_lane_id / 4;
        sm = (qid & 4) + (simd_lane_id / 2) % 4;
        sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
        
        // Compute starting offsets in threadgroup memory
        As_offset = transpose_a 
            ? ((sn) * lda_tgp + (tm + sm))
            : ((sn) + (tm + sm) * lda_tgp);
        Bs_offset = transpose_b
            ? ((tn + sn) * ldb_tgp + (sm))
            : ((sm) * ldb_tgp + (tn + sn));
        
        // Zero-initialize result matrices
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < TM * TN; ++i) {
            results[i] = simdgroup_matrix<AccumType, 8, 8>(0);
        }
    }
    
    /// Perform matrix multiply-accumulate for one BK tile.
    /// Uses thread_elements() for loading (MLX pattern).
    METAL_FUNC void mma(
        const threadgroup T* As,
        const threadgroup T* Bs
    ) {
        // Adjust for simdgroup and thread location
        As += As_offset;
        Bs += Bs_offset;
        
        // Iterate over BK in 8-element chunks
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < BK; kk += 8) {
            simdgroup_barrier(mem_flags::mem_none);
            
            // Load A tiles using thread_elements() (MLX pattern)
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < TM; ++i) {
                Asimd[i].thread_elements()[0] = 
                    static_cast<AccumType>(As[i * simd_stride_a + 0]);
                Asimd[i].thread_elements()[1] = 
                    static_cast<AccumType>(As[i * simd_stride_a + jump_a]);
            }
            
            simdgroup_barrier(mem_flags::mem_none);
            
            // Load B tiles using thread_elements() (MLX pattern)
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < TN; ++j) {
                Bsimd[j].thread_elements()[0] = 
                    static_cast<AccumType>(Bs[j * simd_stride_b + 0]);
                Bsimd[j].thread_elements()[1] = 
                    static_cast<AccumType>(Bs[j * simd_stride_b + jump_b]);
            }
            
            simdgroup_barrier(mem_flags::mem_none);
            
            // Multiply and accumulate (serpentine pattern for better cache)
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < TM; ++i) {
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < TN; ++j) {
                    short j_serp = (i % 2) ? (TN - 1 - j) : j;
                    simdgroup_multiply_accumulate(
                        results[i * TN + j_serp],
                        Asimd[i],
                        Bsimd[j_serp],
                        results[i * TN + j_serp]
                    );
                }
            }
            
            // Progress to next K tile
            As += tile_stride_a;
            Bs += tile_stride_b;
        }
    }
    
    /// Store accumulated results to device memory.
    /// Uses thread_elements() for reading (MLX pattern).
    METAL_FUNC void store_result(device U* D, const int ldd) const {
        // Adjust for simdgroup and thread location
        D += (sm + tm) * ldd + tn + sn;
        
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < TM; ++i) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < TN; ++j) {
                thread const auto& accum = results[i * TN + j].thread_elements();
                int offset = (i * TM_stride) * ldd + (j * TN_stride);
                
                D[offset] = static_cast<U>(accum[0]);
                D[offset + 1] = static_cast<U>(accum[1]);
            }
        }
    }
    
    /// Store results with bounds checking (for edge tiles).
    METAL_FUNC void store_result_safe(
        device U* D,
        const int ldd,
        short2 dst_tile_dims
    ) const {
        // Adjust for simdgroup and thread location
        D += (sm + tm) * ldd + tn + sn;
        
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < TM; ++i) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < TN; ++j) {
                thread const auto& accum = results[i * TN + j].thread_elements();
                int offset = (i * TM_stride) * ldd + (j * TN_stride);
                
                // Check bounds for each element
                short row0 = sm + tm + i * TM_stride;
                short col0 = sn + tn + j * TN_stride;
                
                if (row0 < dst_tile_dims.y && col0 < dst_tile_dims.x) {
                    D[offset] = static_cast<U>(accum[0]);
                }
                if (row0 < dst_tile_dims.y && (col0 + 1) < dst_tile_dims.x) {
                    D[offset + 1] = static_cast<U>(accum[1]);
                }
            }
        }
    }
    
    /// Apply alpha scaling to accumulated results.
    METAL_FUNC void apply_alpha(float alpha) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < TM * TN; ++i) {
            thread auto& accum = results[i].thread_elements();
            accum[0] *= alpha;
            accum[1] *= alpha;
        }
    }
    
    /// Apply alpha*result + beta*C epilogue.
    METAL_FUNC void apply_epilogue(
        const device U* C,
        const int ldc,
        float alpha,
        float beta
    ) {
        if (beta == 0.0f) {
            apply_alpha(alpha);
            return;
        }
        
        // Adjust C pointer for thread location
        C += (sm + tm) * ldc + tn + sn;
        
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < TM; ++i) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < TN; ++j) {
                thread auto& accum = results[i * TN + j].thread_elements();
                int offset = (i * TM_stride) * ldc + (j * TN_stride);
                
                // Load C values and apply epilogue
                AccumType c0 = static_cast<AccumType>(C[offset]);
                AccumType c1 = static_cast<AccumType>(C[offset + 1]);
                
                accum[0] = alpha * accum[0] + beta * c0;
                accum[1] = alpha * accum[1] + beta * c1;
            }
        }
    }

    /// Apply epilogue with bounds checking for C load.
    METAL_FUNC void apply_epilogue_safe(
        const device U* C,
        const int ldc,
        float alpha,
        float beta,
        short2 src_tile_dims
    ) {
        if (beta == 0.0f) {
            apply_alpha(alpha);
            return;
        }
        
        // Adjust C pointer for thread location
        C += (sm + tm) * ldc + tn + sn;
        
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < TM; ++i) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < TN; ++j) {
                thread auto& accum = results[i * TN + j].thread_elements();
                int offset = (i * TM_stride) * ldc + (j * TN_stride);
                
                // Check bounds for C load
                short row0 = sm + tm + i * TM_stride;
                short col0 = sn + tn + j * TN_stride;
                
                AccumType c0 = 0.0f;
                AccumType c1 = 0.0f;
                
                if (row0 < src_tile_dims.y && col0 < src_tile_dims.x) {
                    c0 = static_cast<AccumType>(C[offset]);
                }
                if (row0 < src_tile_dims.y && (col0 + 1) < src_tile_dims.x) {
                    c1 = static_cast<AccumType>(C[offset + 1]);
                }
                
                accum[0] = alpha * accum[0] + beta * c0;
                accum[1] = alpha * accum[1] + beta * c1;
            }
        }
    }
    
    /// Apply activation to accumulated results.
    template<typename Activation = ActivationNone>
    METAL_FUNC void apply_activation() {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < TM * TN; ++i) {
            thread auto& accum = results[i].thread_elements();
            accum[0] = Activation::apply(accum[0]);
            accum[1] = Activation::apply(accum[1]);
        }
    }
    
    /// Apply row-wise bias.
    METAL_FUNC void apply_bias(const device T* bias, const int col_offset) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < TM; ++i) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < TN; ++j) {
                thread auto& accum = results[i * TN + j].thread_elements();
                
                // Bias is indexed by column
                short col0 = sn + tn + j * TN_stride + col_offset;
                short col1 = col0 + 1;
                
                accum[0] += static_cast<AccumType>(bias[col0]);
                accum[1] += static_cast<AccumType>(bias[col1]);
            }
        }
    }
};

#endif // METALLIC_MMA_METAL
