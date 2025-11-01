#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

// Parameters for the tiled GEMM kernel
struct GemmTiledParams {
    uint m;
    uint n;
    uint k;
    uint lda;  // leading dimension of A
    uint ldb;  // leading dimension of B
    uint ldc;  // leading dimension of C
    uint tile_m;  // tile size for M dimension
    uint tile_n;  // tile size for N dimension
    uint tile_k;  // tile size for K dimension
    uint use_simdgroup_mm; // 0 when SIMD-group MMA is unavailable
    float alpha;
    float beta;
};

// Tiled GEMM kernel using SIMD group matrix multiply (if device supports it)
// Falls back to threadgroup memory approach if hardware matrix multiply not available
kernel void gemm_tiled_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant const GemmTiledParams* params [[buffer(3)]],
    device const half* bias [[buffer(4)]],  // optional bias buffer
    threadgroup half* tg_a [[threadgroup(0)]],
    threadgroup half* tg_b [[threadgroup(1)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]
) {
    const uint m = params->m;
    const uint n = params->n;
    const uint k = params->k;
    const uint lda = params->lda;
    const uint ldb = params->ldb;
    const uint ldc = params->ldc;
    const uint tile_m = params->tile_m;
    const uint tile_n = params->tile_n;
    const uint tile_k = params->tile_k;
    const float alpha = params->alpha;
    const float beta = params->beta;
    
    const uint tg_x = gid.x;
    const uint tg_y = gid.y;
    
    // Calculate global output position
    const uint global_m = tg_y * tile_m;
    const uint global_n = tg_x * tile_n;
    
    // SIMD group dimensions - use 8x8 which is well-optimized on Apple Silicon
    // For larger effective tiles, we'll use multiple simdgroups per operation
    constexpr const uint simdgroup_m = 8;
    constexpr const uint simdgroup_n = 8;
    
    // Threadgroup dimensions - depends on how many 8x8 simdgroups we want per threadgroup
    // For a 32-thread simdgroup, we might use 64 threads = 2 simdgroups per threadgroup
    constexpr const uint threads_per_simdgroup = 32;
    constexpr const uint simdgroups_per_threadgroup = 2;
    constexpr const uint threads_per_threadgroup = threads_per_simdgroup * simdgroups_per_threadgroup; // 64
    
#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 300) && defined(__HAVE_SIMDGROUP_MATRIX__)
    // Use hardware accelerated matrix multiply when the device supports SIMD-group MMA.
    if (params->use_simdgroup_mm != 0 && tile_m % simdgroup_m == 0 && tile_n % simdgroup_n == 0) {
        // Define matrix types for simdgroup operations - use 8x8 which is optimized for Apple Silicon
        using a_matrix = simdgroup_matrix<half, 8, 8>;
        using b_matrix = simdgroup_matrix<half, 8, 8>;
        using c_matrix = simdgroup_matrix<float, 8, 8>;
        
        // Calculate the number of simdgroups needed in each dimension
        const uint simdgroups_m = tile_m / simdgroup_m;
        const uint simdgroups_n = tile_n / simdgroup_n;
        
        // Create result accumulator
        const uint local_simdgroup_id = simdgroup_id;
        const uint local_simdgroup_m = local_simdgroup_id / simdgroups_n;
        const uint local_simdgroup_n = local_simdgroup_id % simdgroups_n;
        
        c_matrix result_accumulator = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        
        // Loop over K dimension in tiles
        for (uint k_iter = 0; k_iter < k; k_iter += tile_k) {
            // Load A tile to shared memory with vectorized access - organized for simdgroup operations
            const uint a_tile_m = min(tile_m, m - global_m);
            const uint a_tile_k = min(tile_k, k - k_iter);
            
            // Load A tile in a way that's compatible with column-major layout expected by matrix_a
            const uint tg_load_size_a = a_tile_m * a_tile_k;
            
            // Load A tile with coalesced access pattern for better performance
            for (uint idx = lid; idx < tg_load_size_a; idx += threads_per_threadgroup) {
                const uint row = idx / a_tile_k;
                const uint col = idx % a_tile_k;
                const uint global_k_pos = k_iter + col;
                const uint global_row_pos = global_m + row;
                
                if (global_row_pos < m && global_k_pos < k) {
                    const uint src_idx = global_row_pos * lda + global_k_pos;
                    tg_a[idx] = A[src_idx];
                } else {
                    // Zero-pad out-of-bounds elements
                    tg_a[idx] = (half)0.0h;
                }
            }
            
            // Load B tile to shared memory with vectorized access - organized for simdgroup operations
            const uint b_tile_k = min(tile_k, k - k_iter);
            const uint b_tile_n = min(tile_n, n - global_n);
            
            const uint tg_load_size_b = b_tile_k * b_tile_n;
            
            // Load B tile with coalesced access pattern for better performance
            for (uint idx = lid; idx < tg_load_size_b; idx += threads_per_threadgroup) {
                const uint row = idx / b_tile_n;
                const uint col = idx % b_tile_n;
                const uint global_k_pos = k_iter + row;
                const uint global_col_pos = global_n + col;
                
                if (global_k_pos < k && global_col_pos < n) {
                    const uint src_idx = global_k_pos * ldb + global_col_pos;
                    tg_b[idx] = B[src_idx];
                } else {
                    // Zero-pad out-of-bounds elements
                    tg_b[idx] = (half)0.0h;
                }
            }
            
            // Synchronize to ensure all data is loaded
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Perform matrix multiply with simdgroup operations
            for (uint k_tile = 0; k_tile < tile_k; k_tile += simdgroup_m) {
                // Prepare matrices for multiplication
                a_matrix a_frag;
                b_matrix b_frag;
                
                // Calculate the position in threadgroup memory for this SIMD group's fragment
                // For simdgroup_load to work properly, the data must be arranged in the correct layout
                threadgroup const half* a_base_ptr = tg_a + (local_simdgroup_m * simdgroup_m) * tile_k + k_tile;
                threadgroup const half* b_base_ptr = tg_b + (k_tile * tile_n) + (local_simdgroup_n * simdgroup_n);
                
                // Load A fragment using simdgroup_load - this properly formats the data for hardware multiply
                simdgroup_load(a_frag, a_base_ptr, tile_k);
                
                // Load B fragment using simdgroup_load - this properly formats the data for hardware multiply
                simdgroup_load(b_frag, b_base_ptr, tile_n);

                // Multiply and accumulate using hardware acceleration
                simdgroup_multiply_accumulate(result_accumulator, a_frag, b_frag, result_accumulator);
            }
            
            // Synchronize after simdgroup operations
            threadgroup_barrier(mem_flags::mem_none);
        }
        
        // Write results to global memory with simple, correct approach - let compiler optimize
        const auto accum_elements = result_accumulator.thread_elements();
        
        for (uint i = 0; i < simdgroup_m; i++) {
            for (uint j = 0; j < simdgroup_n; j++) {
                uint global_m_idx = global_m + local_simdgroup_m * simdgroup_m + i;
                uint global_n_idx = global_n + local_simdgroup_n * simdgroup_n + j;
                
                if (global_m_idx < m && global_n_idx < n) {
                    uint c_idx = global_m_idx * ldc + global_n_idx;
                    float current_val = (float)C[c_idx];
                    half bias_val = (bias != nullptr) ? bias[global_n_idx] : (half)0.0f;
                    const float accum_val = accum_elements[i * simdgroup_n + j];
                    C[c_idx] = (half)(alpha * accum_val + beta * current_val + (float)bias_val);
                }
            }
        }
        return;
    }
#endif // defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 300) && defined(__HAVE_SIMDGROUP_MATRIX__)
    // Fallback path using threadgroup memory and software matrix multiply
    // Define threadgroup tile sizes to reduce register pressure and improve occupancy
    // Use 8x8 tiles to reduce register usage (64 floats vs 256 floats per thread)
    constexpr const uint tg_tile_m = 8;
    constexpr const uint tg_tile_n = 8;
    constexpr const uint tg_tile_k = 16;
        
        // Allocate shared memory tiles
        threadgroup half a_tile[tg_tile_m * tg_tile_k];
        threadgroup half b_tile[tg_tile_k * tg_tile_n];
        
        // Calculate threadgroup coordinates for 8x8 tiles
        // Use 64 threads per threadgroup (8x8), with each thread handling 1 output element
        const uint threads_per_tg = 64;  // Use 64 threads per threadgroup (8x8)
        const uint tg_thread_x = lid % 8;
        const uint tg_thread_y = lid / 8;
        
        // Calculate global output position
        const uint global_m_start = tg_y * tg_tile_m;
        const uint global_n_start = tg_x * tg_tile_n;
        
        // Each thread handles one output element, not the entire 8x8 tile
        const uint my_m = tg_thread_y;
        const uint my_n = tg_thread_x;
        float accumulator = 0.0f;  // Single float, not array
        
        // Loop over K dimension in tiles
        for (uint k_iter = 0; k_iter < k; k_iter += tg_tile_k) {
            // Load A tile to shared memory with simple linear access
            for (uint base_ti = lid; base_ti < tg_tile_m * tg_tile_k; base_ti += threads_per_tg) {
                uint ti = base_ti;
                uint local_m = ti / tg_tile_k;
                uint local_k = ti % tg_tile_k;
                uint global_m = global_m_start + local_m;
                uint global_k = k_iter + local_k;
                
                if (global_m < m && global_k < k) {
                    uint src_idx = global_m * lda + global_k;
                    a_tile[ti] = A[src_idx];
                } else {
                    a_tile[ti] = (half)0.0h;
                }
            }
            
            // Load B tile to shared memory with simple linear access
            for (uint base_ti = lid; base_ti < tg_tile_k * tg_tile_n; base_ti += threads_per_tg) {
                uint ti = base_ti;
                uint local_k = ti / tg_tile_n;
                uint local_n = ti % tg_tile_n;
                uint global_k = k_iter + local_k;
                uint global_n = global_n_start + local_n;
                
                if (global_k < k && global_n < n) {
                    uint src_idx = global_k * ldb + global_n;
                    b_tile[ti] = B[src_idx];
                } else {
                    b_tile[ti] = (half)0.0h;
                }
            }
            
            // Synchronize to ensure all data is loaded
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Perform the computation for this K tile
            // Each thread computes its own accumulator cell by processing the corresponding row of A and column of B
            #pragma unroll
            for (uint k = 0; k < tg_tile_k; k++) {
                // Load the A value for this thread's row
                half a_val = a_tile[my_m * tg_tile_k + k];
                float a_f32 = (float)a_val;
                
                // Load the B value for this thread's column
                half b_val = b_tile[k * tg_tile_n + my_n];
                float b_f32 = (float)b_val;
                
                // Accumulate the contribution of this k-index
                accumulator += a_f32 * b_f32;
            }
            
            // Synchronize after computation for this K tile
            threadgroup_barrier(mem_flags::mem_none);
        }
        
        // Write results back to global memory with improved vectorized stores
        // Each thread writes its own result
        uint output_global_m = global_m_start + my_m;
        uint output_global_n = global_n_start + my_n;
        
        if (output_global_m < m && output_global_n < n) {
            uint c_idx = output_global_m * ldc + output_global_n;
            float current_val = (float)C[c_idx];
            half bias_val = (bias != nullptr) ? bias[output_global_n] : (half)0.0f;
            half result = (half)(alpha * accumulator + beta * current_val + (float)bias_val);
            C[c_idx] = result;
        }
}
