// Dense GEMV kernels

// Optimized SIMD-Parallel FP16 GEMV
// Ports the logic from Q8 (Vectorized 4-Block + 2x Unroll) to F16.
// Layout: Column-Major (standard).



// -----------------------------------------------------------------------------
// Small-N GEMV kernels for dense FP16 matmuls (N ∈ {1,2,4,8,16})
// -----------------------------------------------------------------------------

// Optimized N=8 GEMV kernel following GGML patterns
// Each thread handles one element of the output matrix C
kernel void gemv_n8_f16(
    device const half* A [[buffer(0)]], // M x K
    device const half* B [[buffer(1)]], // K x 8
    device half*       C [[buffer(2)]], // M x 8
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup half smemB[GEMV_SMALLN_TILE_K * 8u];
    gemv_f16_smalln_impl<8u, 8u, GEMV_SMALLN_TILE_K, false>(A, B, C, M, K, tid, tgid, smemB);
}

// N=4 GEMV kernel - optimized for small N=4 case
kernel void gemv_n4_f16(
    device const half* A [[buffer(0)]], // M x K
    device const half* B [[buffer(1)]], // K x 4
    device half*       C [[buffer(2)]], // M x 4
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup half smemB[GEMV_SMALLN_TILE_K * 4u];
    // This variant historically advanced rows along tgid.y; preserve that behavior (UseGridY=true).
    gemv_f16_smalln_impl<4u, 16u, GEMV_SMALLN_TILE_K, true>(A, B, C, M, K, tid, tgid, smemB);
}

// N=1 GEMV kernel - optimized for single column output
kernel void gemv_n1_f16(
    device const half* A [[buffer(0)]], // M x K
    device const half* B [[buffer(1)]], // K x 1
    device half*       C [[buffer(2)]], // M x 1
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup half smemB[GEMV_SMALLN_TILE_K * 1u];
    gemv_f16_smalln_impl<1u, 32u, GEMV_SMALLN_TILE_K, false>(A, B, C, M, K, tid, tgid, smemB);
}

// N=2 GEMV kernel - optimized for N=2 case
kernel void gemv_n2_f16(
    device const half* A [[buffer(0)]], // M x K
    device const half* B [[buffer(1)]], // K x 2
    device half*       C [[buffer(2)]], // M x 2
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup half smemB[GEMV_SMALLN_TILE_K * 2u];
    gemv_f16_smalln_impl<2u, 32u, GEMV_SMALLN_TILE_K, false>(A, B, C, M, K, tid, tgid, smemB);
}

// N=16 GEMV kernel - optimized for N=16 case
kernel void gemv_n16_f16(
    device const half* A [[buffer(0)]], // M x K
    device const half* B [[buffer(1)]], // K x 16
    device half*       C [[buffer(2)]], // M x 16
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup half smemB[GEMV_SMALLN_TILE_K * 16u];
    gemv_f16_smalln_impl<16u, 4u, GEMV_SMALLN_TILE_K, false>(A, B, C, M, K, tid, tgid, smemB);
}



