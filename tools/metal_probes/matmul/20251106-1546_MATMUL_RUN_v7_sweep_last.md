Enhanced variants configuration available, enhanced Swift code will use it
Compiling m1_dot_product_v2.metal → m1_dot_product_v2.metallib
Compiling m1_dot_product_v3.metal → m1_dot_product_v3.metallib
Compiling m1_dot_product_v4.metal → m1_dot_product_v4.metallib
Compiling m1_dot_product_v5.metal → m1_dot_product_v5.metallib
Compiling m1_dot_product_v6.metal → m1_dot_product_v6.metallib
Compiling m1_dot_product_v7.metal → m1_dot_product_v7.metallib
Compiling m1_dot_product.metal → m1_dot_product.metallib
Compiling optimized_mlx_bk64.metal → optimized_mlx_bk64.metallib
Compiling optimized_mlx_m1.metal → optimized_mlx_m1.metallib
Compiling optimized_mlx_nn.metal → optimized_mlx_nn.metallib
Compiling original_gemm_tiled.metal → original_gemm_tiled.metallib
Compiling original_gemv_smalln.metal → original_gemv_smalln.metallib
Compiling original_gemv.metal → original_gemv.metallib
Compiling original_mlx.metal → original_mlx.metallib
Enhanced matmul probe compilation succeeded.
Executing enhanced matmul probe harness...
Pre-computing or loading 20 unique CPU reference tensors...
Loaded tensor from cache: spec_matmul_alpha_beta_mlx_m1_n64_k67_tA0_tB0_alpha1_0_beta0_0_bias0_batch14.bin
Loaded tensor from cache: spec_matmul_alpha_beta_mlx_m1_n64_k83_tA0_tB0_alpha1_0_beta0_0_bias0_batch14.bin
Loaded tensor from cache: spec_matmul_mps_m1_n9728_k896_tA0_tB1_alpha1_0_beta0_0_bias0_batch1.bin
Loaded tensor from cache: spec_matmul_alpha_beta_mlx_m1_n73_k64_tA0_tB1_alpha1_0_beta0_0_bias0_batch14.bin
Loaded tensor from cache: spec_matmul_bias_add_mlx_m1_n1152_k896_tA0_tB0_alpha1_0_beta0_0_bias0_batch1.bin
Loaded tensor from cache: spec_matmul_alpha_beta_mlx_m1_n64_k68_tA0_tB0_alpha1_0_beta0_0_bias0_batch14.bin
Loaded tensor from cache: spec_matmul_alpha_beta_mlx_m1_n51_k64_tA0_tB1_alpha1_0_beta0_0_bias0_batch14.bin
Loaded tensor from cache: spec_matmul_mlx_m1_n896_k896_tA0_tB1_alpha1_0_beta0_0_bias0_batch1.bin
Loaded tensor from cache: spec_matmul_alpha_beta_mlx_m1_n76_k64_tA0_tB1_alpha1_0_beta0_0_bias0_batch14.bin
Loaded tensor from cache: spec_matmul_mps_m1_n896_k4864_tA0_tB1_alpha1_0_beta0_0_bias0_batch1.bin
Loaded tensor from cache: spec_matmul_alpha_beta_mlx_m1_n64_k95_tA0_tB0_alpha1_0_beta0_0_bias0_batch14.bin
Loaded tensor from cache: spec_matmul_alpha_beta_mlx_m1_n80_k64_tA0_tB1_alpha1_0_beta0_0_bias0_batch14.bin
Loaded tensor from cache: spec_matmul_mps_m1_n151936_k896_tA0_tB1_alpha1_0_beta0_0_bias0_batch1.bin
Loaded tensor from cache: spec_matmul_alpha_beta_mlx_m1_n64_k69_tA0_tB0_alpha1_0_beta0_0_bias0_batch14.bin
Loaded tensor from cache: spec_matmul_alpha_beta_mlx_m1_n67_k64_tA0_tB1_alpha1_0_beta0_0_bias0_batch14.bin
Loaded tensor from cache: spec_matmul_alpha_beta_mlx_m1_n64_k84_tA0_tB0_alpha1_0_beta0_0_bias0_batch14.bin
Loaded tensor from cache: spec_matmul_alpha_beta_mlx_m1_n64_k88_tA0_tB0_alpha1_0_beta0_0_bias0_batch14.bin
Loaded tensor from cache: spec_matmul_alpha_beta_mlx_m1_n64_k16_tA0_tB0_alpha1_0_beta0_0_bias0_batch14.bin
Loaded tensor from cache: spec_matmul_alpha_beta_mlx_m1_n83_k64_tA0_tB1_alpha1_0_beta0_0_bias0_batch14.bin
Loaded tensor from cache: spec_matmul_alpha_beta_mlx_m1_n90_k64_tA0_tB1_alpha1_0_beta0_0_bias0_batch14.bin
Tensor pre-computation and loading complete.
Processing variant: m1_optimized_v2/nt_bn64_col_vec4_bk64_tg128
  All results for m1_optimized_v2/nt_bn64_col_vec4_bk64_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v6/nt_bn32_largek_smalln_kpar_bk128_tg128
  No supported specs for m1_optimized_v6/nt_bn32_largek_smalln_kpar_bk128_tg128.
Processing variant: m1_optimized_v4/nt_tiny_bn32_tg64
  All results for m1_optimized_v4/nt_tiny_bn32_tg64 are cached. Skipping benchmark run.
Processing variant: mlx/m1_fast_tt
  No supported specs for mlx/m1_fast_tt.
Processing variant: m1_optimized_v2/nt_bn128_col_bk64_tg64
  All results for m1_optimized_v2/nt_bn128_col_bk64_tg64 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v6/nt_bn64_col_vec4_tgread_vA_bk64_tg128
  All results for m1_optimized_v6/nt_bn64_col_vec4_tgread_vA_bk64_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v7/nt_bn8_largek_smalln_hybrid_kpar_bk256_tg256
  No supported specs for m1_optimized_v7/nt_bn8_largek_smalln_hybrid_kpar_bk256_tg256.
Processing variant: m1_optimized_v6/nt_bn4_largek_smalln_kpar_bk128_tg128
  No supported specs for m1_optimized_v6/nt_bn4_largek_smalln_kpar_bk128_tg128.
Processing variant: mlx/baseline
  All results for mlx/baseline are cached. Skipping benchmark run.
Processing variant: mlx/m1_fast_nt
  All results for mlx/m1_fast_nt are cached. Skipping benchmark run.
Processing variant: m1_optimized_v6/nt_bn32_largek_smalln_kpar_bk64_tg128
  No supported specs for m1_optimized_v6/nt_bn32_largek_smalln_kpar_bk64_tg128.
Processing variant: m1_optimized_v6/nt_bn16_largek_smalln_kpar_bk64_tg128
  No supported specs for m1_optimized_v6/nt_bn16_largek_smalln_kpar_bk64_tg128.
Processing variant: m1_optimized_v5/nt_bn4_largek_smalln_tg256
  No supported specs for m1_optimized_v5/nt_bn4_largek_smalln_tg256.
Processing variant: m1_optimized_v5/nt_bn16_largek_smalln_tg256
  No supported specs for m1_optimized_v5/nt_bn16_largek_smalln_tg256.
Processing variant: m1_optimized_v7/nt_bn64_col_vec4_worksteal_chunk4_bk128_tg128
  All results for m1_optimized_v7/nt_bn64_col_vec4_worksteal_chunk4_bk128_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v7/nt_bn16_largek_smalln_kpar_vec4_both_bk256_tg256
  No supported specs for m1_optimized_v7/nt_bn16_largek_smalln_kpar_vec4_both_bk256_tg256.
Processing variant: m1_optimized_v5/nt_ultra_tiny_single_tg32
  All results for m1_optimized_v5/nt_ultra_tiny_single_tg32 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v6/nt_bn224_col_vec4_tgread_vA_bk128_tg128
  All results for m1_optimized_v6/nt_bn224_col_vec4_tgread_vA_bk128_tg128 are cached. Skipping benchmark run.
Processing variant: mlx/m1_fast_nn
  All results for mlx/m1_fast_nn are cached. Skipping benchmark run.
Processing variant: m1_optimized_v6/nt_bn192_col_vec4_tgread_vA_bk128_tg128
  All results for m1_optimized_v6/nt_bn192_col_vec4_tgread_vA_bk128_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v6/nt_bn64_largek_smalln_kpar_bk256_tg256
  No supported specs for m1_optimized_v6/nt_bn64_largek_smalln_kpar_bk256_tg256.
Processing variant: m1_optimized_v7/nt_bn256_fused_bias_relu_tg256
  No supported specs for m1_optimized_v7/nt_bn256_fused_bias_relu_tg256.
Processing variant: m1_optimized_v6/nt_bn192_col_vec4_tgread_vA_bk64_tg128
  All results for m1_optimized_v6/nt_bn192_col_vec4_tgread_vA_bk64_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v6/nt_bn224_col_vec4_tgread_vA_bk64_tg128
  All results for m1_optimized_v6/nt_bn224_col_vec4_tgread_vA_bk64_tg128 are cached. Skipping benchmark run.
Processing variant: mlx/bk64_nn
  All results for mlx/bk64_nn are cached. Skipping benchmark run.
Processing variant: m1_optimized_v6/nt_bn4_largek_smalln_kpar_bk256_tg256
  No supported specs for m1_optimized_v6/nt_bn4_largek_smalln_kpar_bk256_tg256.
Processing variant: m1_optimized_v7/nt_bn16_smalln_simdgroupmm_tg256
  No supported specs for m1_optimized_v7/nt_bn16_smalln_simdgroupmm_tg256.
Processing variant: m1_optimized_v6/nt_bn16_largek_smalln_kpar_bk128_tg128
  No supported specs for m1_optimized_v6/nt_bn16_largek_smalln_kpar_bk128_tg128.
Processing variant: m1_optimized_v2/nt_bn64_col_bk64_tg64
  All results for m1_optimized_v2/nt_bn64_col_bk64_tg64 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v4/nt_bn8_largek_smalln_tg128
  No supported specs for m1_optimized_v4/nt_bn8_largek_smalln_tg128.
Processing variant: m1_optimized_v6/nt_bn256_col_vec4_tgread_vA_bk64_tg256
  All results for m1_optimized_v6/nt_bn256_col_vec4_tgread_vA_bk64_tg256 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v3/nt_bn64_col_vec4_tgread_bk128_tg128
  All results for m1_optimized_v3/nt_bn64_col_vec4_tgread_bk128_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v7/nt_bn128_col_vec4_worksteal_bk128_tg128
  All results for m1_optimized_v7/nt_bn128_col_vec4_worksteal_bk128_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v5/nt_bn8_largek_smalln_tg512
  No supported specs for m1_optimized_v5/nt_bn8_largek_smalln_tg512.
Processing variant: m1_optimized_v7/nt_bn128_col_vec4_triplebuf_bk64_tg128
  All results for m1_optimized_v7/nt_bn128_col_vec4_triplebuf_bk64_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v7/nt_bn8_largek_smalln_kpar_vec4_both_bk256_tg256
  No supported specs for m1_optimized_v7/nt_bn8_largek_smalln_kpar_vec4_both_bk256_tg256.
Processing variant: m1_optimized_v5/nt_bn16_largek_smalln_tg512
  No supported specs for m1_optimized_v5/nt_bn16_largek_smalln_tg512.
Processing variant: m1_optimized_v3/nt_bn128_col_vec4_tgread_bk64_tg128
  All results for m1_optimized_v3/nt_bn128_col_vec4_tgread_bk64_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v7/nt_adaptive_bn128_tg128
  All results for m1_optimized_v7/nt_adaptive_bn128_tg128 are cached. Skipping benchmark run.
Processing variant: mlx/baseline_nn
  All results for mlx/baseline_nn are cached. Skipping benchmark run.
Processing variant: mlx/m1_fast_tn
  No supported specs for mlx/m1_fast_tn.
Processing variant: m1_optimized_v5/nt_bn4_largek_smalln_tg512
  No supported specs for m1_optimized_v5/nt_bn4_largek_smalln_tg512.
Processing variant: m1_optimized_v6/nt_bn256_col_vec4_tgread_vA_bk128_tg128
  All results for m1_optimized_v6/nt_bn256_col_vec4_tgread_vA_bk128_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v6/nt_bn64_col_vec4_tgread_vA_bk128_tg128
  All results for m1_optimized_v6/nt_bn64_col_vec4_tgread_vA_bk128_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v4/nt_bn16_largek_smalln_tg256
  No supported specs for m1_optimized_v4/nt_bn16_largek_smalln_tg256.
Processing variant: m1_optimized_v6/nt_bn8_largek_smalln_kpar_bk64_tg128
  No supported specs for m1_optimized_v6/nt_bn8_largek_smalln_kpar_bk64_tg128.
Processing variant: gemv/baseline
  All results for gemv/baseline are cached. Skipping benchmark run.
Processing variant: m1_optimized_v2/nt_bn128_col_vec4_bk128_tg128
  All results for m1_optimized_v2/nt_bn128_col_vec4_bk128_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v7/nt_bn64_col_vec4_transformB_bk64_tg128
  All results for m1_optimized_v7/nt_bn64_col_vec4_transformB_bk64_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v4/nt_bn256_col_vec4_tgread_bk64_tg128
  All results for m1_optimized_v4/nt_bn256_col_vec4_tgread_bk64_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v2/nt_bn128_col_bk128_tg128
  All results for m1_optimized_v2/nt_bn128_col_bk128_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v4/nt_tiny_bn64_tg64
  All results for m1_optimized_v4/nt_tiny_bn64_tg64 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v2/nt_bn64_col_bk64_tg128
  All results for m1_optimized_v2/nt_bn64_col_bk64_tg128 are cached. Skipping benchmark run.
Processing variant: mlx/bk64_nt
  All results for mlx/bk64_nt are cached. Skipping benchmark run.
Processing variant: mlx/nn_wide
  All results for mlx/nn_wide are cached. Skipping benchmark run.
Processing variant: mlx/bk64_tt
  No supported specs for mlx/bk64_tt.
Processing variant: m1_optimized_v5/nt_bn8_largek_smalln_tg256
  No supported specs for m1_optimized_v5/nt_bn8_largek_smalln_tg256.
Processing variant: m1_optimized_v3/nt_bn128_col_vec4_sgbr_bk64_tg128
  All results for m1_optimized_v3/nt_bn128_col_vec4_sgbr_bk64_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v4/nt_bn8_largek_smalln_tg256
  No supported specs for m1_optimized_v4/nt_bn8_largek_smalln_tg256.
Processing variant: m1_optimized_v6/nt_bn160_col_vec4_tgread_vA_bk64_tg128
  All results for m1_optimized_v6/nt_bn160_col_vec4_tgread_vA_bk64_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v2/nt_bn128_col
  All results for m1_optimized_v2/nt_bn128_col are cached. Skipping benchmark run.
Processing variant: m1_optimized_v6/nt_bn128_col_vec4_tgread_vA_bk128_tg128
  All results for m1_optimized_v6/nt_bn128_col_vec4_tgread_vA_bk128_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v6/nt_bn64_largek_smalln_kpar_bk128_tg128
  No supported specs for m1_optimized_v6/nt_bn64_largek_smalln_kpar_bk128_tg128.
Processing variant: m1_optimized_v6/nt_bn8_largek_smalln_kpar_bk256_tg256
  No supported specs for m1_optimized_v6/nt_bn8_largek_smalln_kpar_bk256_tg256.
Processing variant: mps/baseline
  All results for mps/baseline are cached. Skipping benchmark run.
Processing variant: m1_optimized_v3/nt_bn128_col_vec4_tgread_bk128_tg128
  All results for m1_optimized_v3/nt_bn128_col_vec4_tgread_bk128_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v6/nt_bn256_col_vec4_tgread_vA_bk64_tg128
  All results for m1_optimized_v6/nt_bn256_col_vec4_tgread_vA_bk64_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v6/nt_bn160_col_vec4_tgread_vA_bk128_tg128
  All results for m1_optimized_v6/nt_bn160_col_vec4_tgread_vA_bk128_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v7/nt_bn256_fused_bias_gelu_tg256
  No supported specs for m1_optimized_v7/nt_bn256_fused_bias_gelu_tg256.
Processing variant: m1_optimized_v2/nt_bn64_col_vec4_bk128_tg128
  All results for m1_optimized_v2/nt_bn64_col_vec4_bk128_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v2/nt_bn64_col
  All results for m1_optimized_v2/nt_bn64_col are cached. Skipping benchmark run.
Processing variant: mlx/nn_fast
  All results for mlx/nn_fast are cached. Skipping benchmark run.
Processing variant: mlx/bk64_tn
  No supported specs for mlx/bk64_tn.
Processing variant: m1_optimized_v6/nt_bn32_largek_smalln_kpar_bk256_tg256
  No supported specs for m1_optimized_v6/nt_bn32_largek_smalln_kpar_bk256_tg256.
Processing variant: gemv/smalln
  No supported specs for gemv/smalln.
Processing variant: m1_optimized_v6/nt_bn128_col_vec4_tgread_vA_bk64_tg128
  All results for m1_optimized_v6/nt_bn128_col_vec4_tgread_vA_bk64_tg128 are cached. Skipping benchmark run.
Processing variant: gemm_tiled/baseline
  All results for gemm_tiled/baseline are cached. Skipping benchmark run.
Processing variant: m1_optimized_v4/nt_bn256_col_vec4_tgread_bk128_tg128
  All results for m1_optimized_v4/nt_bn256_col_vec4_tgread_bk128_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v2/nt_bn128_col_bk64_tg128
  All results for m1_optimized_v2/nt_bn128_col_bk64_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v6/nt_bn8_largek_smalln_kpar_bk128_tg128
  No supported specs for m1_optimized_v6/nt_bn8_largek_smalln_kpar_bk128_tg128.
Processing variant: m1_optimized_v4/nt_bn4_largek_smalln_tg256
  No supported specs for m1_optimized_v4/nt_bn4_largek_smalln_tg256.
Processing variant: m1_optimized_v6/nt_bn16_largek_smalln_kpar_bk256_tg256
  No supported specs for m1_optimized_v6/nt_bn16_largek_smalln_kpar_bk256_tg256.
Processing variant: m1_optimized_v2/nt_bn64_col_bk128_tg128
  All results for m1_optimized_v2/nt_bn64_col_bk128_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v2/nt_bn128_col_vec4_bk64_tg128
  All results for m1_optimized_v2/nt_bn128_col_vec4_bk64_tg128 are cached. Skipping benchmark run.
Processing variant: m1_optimized_v7/nt_bn128_col_vec4_worksteal_chunk4_bk128_tg128
  All results for m1_optimized_v7/nt_bn128_col_vec4_worksteal_chunk4_bk128_tg128 are cached. Skipping benchmark run.
Benchmark completed. Processed 247 total results with 1453 failures.
Matmul benchmark summary (iterations=6)
Spec op=matmul | batch=1 | m=1 | n=9728 | k=896 | tA=0 | tB=1:
   - mlx/baseline: avg_gpu=0.149ms | avg_cpu=0.273ms [vs orig 0.325ms] (-16.14%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - mlx/bk64_nt: avg_gpu=0.139ms | avg_cpu=0.252ms [vs orig 0.325ms] (-22.57%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - mlx/m1_fast_nt: avg_gpu=0.152ms | avg_cpu=0.253ms [vs orig 0.325ms] (-22.06%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v2/nt_bn128_col: avg_gpu=0.144ms | avg_cpu=0.289ms [vs orig 0.325ms] (-11.06%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v2/nt_bn128_col_bk128_tg128: avg_gpu=0.136ms | avg_cpu=0.283ms [vs orig 0.325ms] (-12.84%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v2/nt_bn128_col_bk64_tg128: avg_gpu=0.14ms | avg_cpu=0.255ms [vs orig 0.325ms] (-21.46%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v2/nt_bn128_col_bk64_tg64: avg_gpu=0.137ms | avg_cpu=0.285ms [vs orig 0.325ms] (-12.40%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v2/nt_bn128_col_vec4_bk128_tg128: avg_gpu=0.133ms | avg_cpu=0.26ms [vs orig 0.325ms] (-20.07%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v2/nt_bn128_col_vec4_bk64_tg128: avg_gpu=0.142ms | avg_cpu=0.284ms [vs orig 0.325ms] (-12.48%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v2/nt_bn64_col: avg_gpu=0.139ms | avg_cpu=0.292ms [vs orig 0.325ms] (-10.24%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v2/nt_bn64_col_bk128_tg128: avg_gpu=0.148ms | avg_cpu=0.31ms [vs orig 0.325ms] (-4.66%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v2/nt_bn64_col_bk64_tg128: avg_gpu=0.133ms | avg_cpu=0.252ms [vs orig 0.325ms] (-22.46%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v2/nt_bn64_col_bk64_tg64: avg_gpu=0.138ms | avg_cpu=0.27ms [vs orig 0.325ms] (-16.90%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v2/nt_bn64_col_vec4_bk128_tg128: avg_gpu=0.134ms | avg_cpu=0.25ms [vs orig 0.325ms] (-23.18%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v2/nt_bn64_col_vec4_bk64_tg128: avg_gpu=0.133ms | avg_cpu=0.265ms [vs orig 0.325ms] (-18.43%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v3/nt_bn128_col_vec4_sgbr_bk64_tg128 [best-cpu]: avg_gpu=0.135ms | avg_cpu=0.239ms [vs orig 0.325ms] (-26.33%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v3/nt_bn128_col_vec4_tgread_bk128_tg128: avg_gpu=0.134ms | avg_cpu=0.256ms [vs orig 0.325ms] (-21.28%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v3/nt_bn128_col_vec4_tgread_bk64_tg128: avg_gpu=0.135ms | avg_cpu=0.283ms [vs orig 0.325ms] (-12.82%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v3/nt_bn64_col_vec4_tgread_bk128_tg128: avg_gpu=0.134ms | avg_cpu=0.257ms [vs orig 0.325ms] (-21.00%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v4/nt_bn256_col_vec4_tgread_bk128_tg128: avg_gpu=0.198ms | avg_cpu=0.328ms [vs orig 0.325ms] (+0.96%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v4/nt_bn256_col_vec4_tgread_bk64_tg128: avg_gpu=0.142ms | avg_cpu=0.248ms [vs orig 0.325ms] (-23.61%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v4/nt_tiny_bn32_tg64 [best-gpu]: avg_gpu=0.131ms | avg_cpu=0.272ms [vs orig 0.325ms] (-16.45%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v4/nt_tiny_bn64_tg64: avg_gpu=0.133ms | avg_cpu=0.284ms [vs orig 0.325ms] (-12.62%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v6/nt_bn128_col_vec4_tgread_vA_bk128_tg128: avg_gpu=0.157ms | avg_cpu=0.374ms [vs orig 0.325ms] (+15.18%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v6/nt_bn128_col_vec4_tgread_vA_bk64_tg128: avg_gpu=0.135ms | avg_cpu=0.287ms [vs orig 0.325ms] (-11.63%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v6/nt_bn160_col_vec4_tgread_vA_bk128_tg128: avg_gpu=0.148ms | avg_cpu=0.346ms [vs orig 0.325ms] (+6.39%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v6/nt_bn160_col_vec4_tgread_vA_bk64_tg128: avg_gpu=0.14ms | avg_cpu=0.244ms [vs orig 0.325ms] (-25.01%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v6/nt_bn192_col_vec4_tgread_vA_bk128_tg128: avg_gpu=0.145ms | avg_cpu=0.261ms [vs orig 0.325ms] (-19.63%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v6/nt_bn192_col_vec4_tgread_vA_bk64_tg128: avg_gpu=0.141ms | avg_cpu=0.248ms [vs orig 0.325ms] (-23.57%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v6/nt_bn224_col_vec4_tgread_vA_bk128_tg128: avg_gpu=0.138ms | avg_cpu=0.287ms [vs orig 0.325ms] (-11.77%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v6/nt_bn224_col_vec4_tgread_vA_bk64_tg128: avg_gpu=0.138ms | avg_cpu=0.26ms [vs orig 0.325ms] (-19.85%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v6/nt_bn256_col_vec4_tgread_vA_bk128_tg128: avg_gpu=0.145ms | avg_cpu=0.267ms [vs orig 0.325ms] (-17.81%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v6/nt_bn256_col_vec4_tgread_vA_bk64_tg128: avg_gpu=0.142ms | avg_cpu=0.255ms [vs orig 0.325ms] (-21.41%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v6/nt_bn256_col_vec4_tgread_vA_bk64_tg256: avg_gpu=0.141ms | avg_cpu=0.263ms [vs orig 0.325ms] (-19.21%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v6/nt_bn64_col_vec4_tgread_vA_bk128_tg128: avg_gpu=0.136ms | avg_cpu=0.25ms [vs orig 0.325ms] (-23.08%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v6/nt_bn64_col_vec4_tgread_vA_bk64_tg128: avg_gpu=0.136ms | avg_cpu=0.253ms [vs orig 0.325ms] (-22.11%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v7/nt_adaptive_bn128_tg128: avg_gpu=0.161ms | avg_cpu=0.288ms [vs orig 0.325ms] (-11.32%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v7/nt_bn128_col_vec4_triplebuf_bk64_tg128: avg_gpu=0.159ms | avg_cpu=0.287ms [vs orig 0.325ms] (-11.68%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v7/nt_bn128_col_vec4_worksteal_bk128_tg128: avg_gpu=0.156ms | avg_cpu=0.3ms [vs orig 0.325ms] (-7.59%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v7/nt_bn128_col_vec4_worksteal_chunk4_bk128_tg128: avg_gpu=0.307ms | avg_cpu=0.531ms [vs orig 0.325ms] (+63.47%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v7/nt_bn64_col_vec4_transformB_bk64_tg128: avg_gpu=0.188ms | avg_cpu=0.305ms [vs orig 0.325ms] (-6.15%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - m1_optimized_v7/nt_bn64_col_vec4_worksteal_chunk4_bk128_tg128: avg_gpu=0.263ms | avg_cpu=0.451ms [vs orig 0.325ms] (+38.66%) maxAbs=1.5522e-02 maxRel=4.7867e-04
   - mps/baseline [baseline]: avg_gpu=0.191ms | avg_cpu=0.311ms [vs orig 0.325ms] (-4.38%) maxAbs=1.5522e-02 maxRel=4.7867e-04
Spec op=matmul | batch=1 | m=1 | n=896 | k=4864 | tA=0 | tB=1:
   - mlx/baseline: avg_gpu=0.193ms | avg_cpu=0.352ms [vs orig 0.239ms] (+47.33%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - mlx/bk64_nt: avg_gpu=0.12ms | avg_cpu=0.276ms [vs orig 0.239ms] (+15.43%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - mlx/m1_fast_nt: avg_gpu=0.416ms | avg_cpu=0.578ms [vs orig 0.239ms] (+142.02%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v2/nt_bn128_col: avg_gpu=0.555ms | avg_cpu=0.708ms [vs orig 0.239ms] (+196.16%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v2/nt_bn128_col_bk128_tg128: avg_gpu=0.162ms | avg_cpu=0.315ms [vs orig 0.239ms] (+31.76%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v2/nt_bn128_col_bk64_tg128: avg_gpu=0.197ms | avg_cpu=0.331ms [vs orig 0.239ms] (+38.54%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v2/nt_bn128_col_bk64_tg64: avg_gpu=0.325ms | avg_cpu=0.456ms [vs orig 0.239ms] (+90.97%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v2/nt_bn128_col_vec4_bk128_tg128: avg_gpu=0.126ms | avg_cpu=0.247ms [vs orig 0.239ms] (+3.22%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v2/nt_bn128_col_vec4_bk64_tg128: avg_gpu=0.138ms | avg_cpu=0.242ms [vs orig 0.239ms] (+1.33%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v2/nt_bn64_col: avg_gpu=0.42ms | avg_cpu=0.572ms [vs orig 0.239ms] (+139.44%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v2/nt_bn64_col_bk128_tg128: avg_gpu=0.192ms | avg_cpu=0.361ms [vs orig 0.239ms] (+51.05%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v2/nt_bn64_col_bk64_tg128: avg_gpu=0.17ms | avg_cpu=0.326ms [vs orig 0.239ms] (+36.36%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v2/nt_bn64_col_bk64_tg64: avg_gpu=0.17ms | avg_cpu=0.29ms [vs orig 0.239ms] (+21.36%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v2/nt_bn64_col_vec4_bk128_tg128: avg_gpu=0.117ms | avg_cpu=0.241ms [vs orig 0.239ms] (+0.65%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v2/nt_bn64_col_vec4_bk64_tg128: avg_gpu=0.13ms | avg_cpu=0.253ms [vs orig 0.239ms] (+5.83%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v3/nt_bn128_col_vec4_sgbr_bk64_tg128: avg_gpu=0.144ms | avg_cpu=0.261ms [vs orig 0.239ms] (+9.21%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v3/nt_bn128_col_vec4_tgread_bk128_tg128: avg_gpu=0.123ms | avg_cpu=0.271ms [vs orig 0.239ms] (+13.23%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v3/nt_bn128_col_vec4_tgread_bk64_tg128: avg_gpu=0.134ms | avg_cpu=0.281ms [vs orig 0.239ms] (+17.70%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v3/nt_bn64_col_vec4_tgread_bk128_tg128: avg_gpu=0.118ms | avg_cpu=0.274ms [vs orig 0.239ms] (+14.60%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v4/nt_tiny_bn32_tg64: avg_gpu=0.104ms | avg_cpu=0.222ms [vs orig 0.239ms] (-7.14%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v4/nt_tiny_bn64_tg64: avg_gpu=0.104ms | avg_cpu=0.221ms [vs orig 0.239ms] (-7.42%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v6/nt_bn128_col_vec4_tgread_vA_bk128_tg128: avg_gpu=0.195ms | avg_cpu=0.335ms [vs orig 0.239ms] (+39.97%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v6/nt_bn128_col_vec4_tgread_vA_bk64_tg128: avg_gpu=0.211ms | avg_cpu=0.336ms [vs orig 0.239ms] (+40.40%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v6/nt_bn160_col_vec4_tgread_vA_bk128_tg128: avg_gpu=0.299ms | avg_cpu=0.456ms [vs orig 0.239ms] (+90.74%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v6/nt_bn160_col_vec4_tgread_vA_bk64_tg128: avg_gpu=0.283ms | avg_cpu=0.495ms [vs orig 0.239ms] (+107.23%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v6/nt_bn192_col_vec4_tgread_vA_bk128_tg128: avg_gpu=0.248ms | avg_cpu=0.364ms [vs orig 0.239ms] (+52.23%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v6/nt_bn192_col_vec4_tgread_vA_bk64_tg128: avg_gpu=0.266ms | avg_cpu=0.385ms [vs orig 0.239ms] (+61.21%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v6/nt_bn224_col_vec4_tgread_vA_bk128_tg128: avg_gpu=0.254ms | avg_cpu=0.377ms [vs orig 0.239ms] (+57.83%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v6/nt_bn224_col_vec4_tgread_vA_bk64_tg128: avg_gpu=0.27ms | avg_cpu=0.371ms [vs orig 0.239ms] (+55.25%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v6/nt_bn64_col_vec4_tgread_vA_bk128_tg128: avg_gpu=0.183ms | avg_cpu=0.289ms [vs orig 0.239ms] (+20.95%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v6/nt_bn64_col_vec4_tgread_vA_bk64_tg128: avg_gpu=0.214ms | avg_cpu=0.341ms [vs orig 0.239ms] (+42.60%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v7/nt_adaptive_bn128_tg128 [bad-rel]: avg_gpu=0.03ms | avg_cpu=0.178ms [vs orig 0.239ms] (-25.37%) maxAbs=8.2621e+01 maxRel=1.6075e+00
   - m1_optimized_v7/nt_bn128_col_vec4_triplebuf_bk64_tg128: avg_gpu=0.428ms | avg_cpu=0.615ms [vs orig 0.239ms] (+157.32%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v7/nt_bn128_col_vec4_worksteal_bk128_tg128: avg_gpu=0.362ms | avg_cpu=0.509ms [vs orig 0.239ms] (+113.02%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v7/nt_bn128_col_vec4_worksteal_chunk4_bk128_tg128: avg_gpu=1.612ms | avg_cpu=1.782ms [vs orig 0.239ms] (+645.72%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v7/nt_bn64_col_vec4_transformB_bk64_tg128: avg_gpu=0.785ms | avg_cpu=0.958ms [vs orig 0.239ms] (+300.84%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - m1_optimized_v7/nt_bn64_col_vec4_worksteal_chunk4_bk128_tg128: avg_gpu=1.454ms | avg_cpu=1.641ms [vs orig 0.239ms] (+586.67%) maxAbs=2.8069e-02 maxRel=4.5125e-04
   - mps/baseline [baseline] [best-gpu] [best-cpu]: avg_gpu=0.058ms | avg_cpu=0.166ms [vs orig 0.239ms] (-30.74%) maxAbs=2.8069e-02 maxRel=4.5125e-04
Spec op=matmul | batch=1 | m=1 | n=896 | k=896 | tA=0 | tB=1:
   - mlx/baseline [baseline]: avg_gpu=0.036ms | avg_cpu=0.208ms [vs orig 0.235ms] (-11.29%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - mlx/bk64_nt: avg_gpu=0.02ms | avg_cpu=0.111ms [vs orig 0.235ms] (-52.90%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - mlx/m1_fast_nt: avg_gpu=0.024ms | avg_cpu=0.17ms [vs orig 0.235ms] (-27.62%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v2/nt_bn128_col: avg_gpu=0.07ms | avg_cpu=0.188ms [vs orig 0.235ms] (-20.13%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v2/nt_bn128_col_bk128_tg128: avg_gpu=0.03ms | avg_cpu=0.18ms [vs orig 0.235ms] (-23.46%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v2/nt_bn128_col_bk64_tg128: avg_gpu=0.029ms | avg_cpu=0.123ms [vs orig 0.235ms] (-47.83%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v2/nt_bn128_col_bk64_tg64: avg_gpu=0.053ms | avg_cpu=0.214ms [vs orig 0.235ms] (-9.07%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v2/nt_bn128_col_vec4_bk128_tg128: avg_gpu=0.02ms | avg_cpu=0.149ms [vs orig 0.235ms] (-36.58%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v2/nt_bn128_col_vec4_bk64_tg128: avg_gpu=0.021ms | avg_cpu=0.132ms [vs orig 0.235ms] (-43.63%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v2/nt_bn64_col: avg_gpu=0.069ms | avg_cpu=0.191ms [vs orig 0.235ms] (-18.71%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v2/nt_bn64_col_bk128_tg128: avg_gpu=0.027ms | avg_cpu=0.136ms [vs orig 0.235ms] (-42.30%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v2/nt_bn64_col_bk64_tg128: avg_gpu=0.03ms | avg_cpu=0.182ms [vs orig 0.235ms] (-22.56%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v2/nt_bn64_col_bk64_tg64: avg_gpu=0.029ms | avg_cpu=0.153ms [vs orig 0.235ms] (-34.93%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v2/nt_bn64_col_vec4_bk128_tg128: avg_gpu=0.02ms | avg_cpu=0.171ms [vs orig 0.235ms] (-27.30%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v2/nt_bn64_col_vec4_bk64_tg128: avg_gpu=0.021ms | avg_cpu=0.137ms [vs orig 0.235ms] (-41.74%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v3/nt_bn128_col_vec4_sgbr_bk64_tg128: avg_gpu=0.023ms | avg_cpu=0.133ms [vs orig 0.235ms] (-43.36%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v3/nt_bn128_col_vec4_tgread_bk128_tg128: avg_gpu=0.023ms | avg_cpu=0.18ms [vs orig 0.235ms] (-23.61%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v3/nt_bn128_col_vec4_tgread_bk64_tg128: avg_gpu=0.022ms | avg_cpu=0.163ms [vs orig 0.235ms] (-30.66%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v3/nt_bn64_col_vec4_tgread_bk128_tg128: avg_gpu=0.02ms | avg_cpu=0.17ms [vs orig 0.235ms] (-27.83%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v4/nt_tiny_bn32_tg64: avg_gpu=0.017ms | avg_cpu=0.137ms [vs orig 0.235ms] (-41.76%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v4/nt_tiny_bn64_tg64: avg_gpu=0.017ms | avg_cpu=0.138ms [vs orig 0.235ms] (-41.37%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v5/nt_ultra_tiny_single_tg32: avg_gpu=0.27ms | avg_cpu=0.406ms [vs orig 0.235ms] (+72.58%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v6/nt_bn128_col_vec4_tgread_vA_bk128_tg128: avg_gpu=0.031ms | avg_cpu=0.179ms [vs orig 0.235ms] (-23.93%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v6/nt_bn128_col_vec4_tgread_vA_bk64_tg128: avg_gpu=0.033ms | avg_cpu=0.185ms [vs orig 0.235ms] (-21.07%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v6/nt_bn160_col_vec4_tgread_vA_bk128_tg128: avg_gpu=0.04ms | avg_cpu=0.166ms [vs orig 0.235ms] (-29.45%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v6/nt_bn160_col_vec4_tgread_vA_bk64_tg128: avg_gpu=0.042ms | avg_cpu=0.133ms [vs orig 0.235ms] (-43.61%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v6/nt_bn192_col_vec4_tgread_vA_bk128_tg128: avg_gpu=0.04ms | avg_cpu=0.19ms [vs orig 0.235ms] (-19.26%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v6/nt_bn192_col_vec4_tgread_vA_bk64_tg128: avg_gpu=0.102ms | avg_cpu=0.249ms [vs orig 0.235ms] (+5.98%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v6/nt_bn224_col_vec4_tgread_vA_bk128_tg128: avg_gpu=0.041ms | avg_cpu=0.19ms [vs orig 0.235ms] (-19.02%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v6/nt_bn224_col_vec4_tgread_vA_bk64_tg128: avg_gpu=0.043ms | avg_cpu=0.156ms [vs orig 0.235ms] (-33.69%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v6/nt_bn64_col_vec4_tgread_vA_bk128_tg128: avg_gpu=0.03ms | avg_cpu=0.149ms [vs orig 0.235ms] (-36.55%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v6/nt_bn64_col_vec4_tgread_vA_bk64_tg128: avg_gpu=0.033ms | avg_cpu=0.139ms [vs orig 0.235ms] (-40.97%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v7/nt_adaptive_bn128_tg128: avg_gpu=0.073ms | avg_cpu=0.232ms [vs orig 0.235ms] (-1.43%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v7/nt_bn128_col_vec4_triplebuf_bk64_tg128: avg_gpu=0.075ms | avg_cpu=0.233ms [vs orig 0.235ms] (-0.81%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v7/nt_bn128_col_vec4_worksteal_bk128_tg128: avg_gpu=0.066ms | avg_cpu=0.219ms [vs orig 0.235ms] (-6.79%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v7/nt_bn128_col_vec4_worksteal_chunk4_bk128_tg128: avg_gpu=0.236ms | avg_cpu=0.392ms [vs orig 0.235ms] (+66.71%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v7/nt_bn64_col_vec4_transformB_bk64_tg128: avg_gpu=0.129ms | avg_cpu=0.268ms [vs orig 0.235ms] (+13.95%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - m1_optimized_v7/nt_bn64_col_vec4_worksteal_chunk4_bk128_tg128: avg_gpu=0.225ms | avg_cpu=0.42ms [vs orig 0.235ms] (+78.56%) maxAbs=8.7585e-03 maxRel=4.7132e-04
   - mps/baseline [best-gpu] [best-cpu]: avg_gpu=0.008ms | avg_cpu=0.099ms [vs orig 0.235ms] (-57.92%) maxAbs=8.7585e-03 maxRel=4.7132e-04
Spec op=matmul_bias_add | batch=1 | m=1 | n=1152 | k=896 | tA=0 | tB=0:
   - mlx/baseline_nn [baseline]: avg_gpu=0.035ms | avg_cpu=0.137ms [vs orig 0.234ms] (-41.30%) maxAbs=1.4099e-02 maxRel=4.7267e-04
   - mlx/bk64_nn: avg_gpu=0.026ms | avg_cpu=0.18ms [vs orig 0.234ms] (-23.11%) maxAbs=1.4099e-02 maxRel=4.7267e-04
   - mlx/m1_fast_nn [best-gpu]: avg_gpu=0.023ms | avg_cpu=0.175ms [vs orig 0.234ms] (-25.10%) maxAbs=1.4099e-02 maxRel=4.7267e-04
   - mlx/nn_fast [best-cpu]: avg_gpu=0.033ms | avg_cpu=0.132ms [vs orig 0.234ms] (-43.38%) maxAbs=1.4099e-02 maxRel=4.7267e-04
   - mlx/nn_wide: avg_gpu=0.032ms | avg_cpu=0.185ms [vs orig 0.234ms] (-21.08%) maxAbs=1.4099e-02 maxRel=4.7267e-04
   - mps/baseline: avg_gpu=0.05ms | avg_cpu=0.175ms [vs orig 0.234ms] (-25.01%) maxAbs=1.4099e-02 maxRel=4.7267e-04
   - gemv/baseline [bad-rel]: avg_gpu=0.061ms | avg_cpu=0.177ms [vs orig 0.234ms] (-24.34%) maxAbs=2.7041e+04 maxRel=1.9968e+00
   - gemm_tiled/baseline [bad-rel]: avg_gpu=0.008ms | avg_cpu=0.129ms [vs orig 0.234ms] (-44.78%) maxAbs=3.6164e+01 maxRel=1.8316e+00
Spec op=matmul | batch=1 | m=1 | n=151936 | k=896 | tA=0 | tB=1:
   - mlx/baseline: avg_gpu=2.174ms | avg_cpu=2.385ms [vs orig 2.51ms] (-4.97%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - mlx/bk64_nt: avg_gpu=2.213ms | avg_cpu=2.46ms [vs orig 2.51ms] (-1.98%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - mlx/m1_fast_nt: avg_gpu=2.053ms | avg_cpu=2.229ms [vs orig 2.51ms] (-11.21%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v2/nt_bn128_col: avg_gpu=3.151ms | avg_cpu=3.347ms [vs orig 2.51ms] (+33.34%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v2/nt_bn128_col_bk128_tg128: avg_gpu=2.094ms | avg_cpu=2.282ms [vs orig 2.51ms] (-9.07%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v2/nt_bn128_col_bk64_tg128: avg_gpu=2.634ms | avg_cpu=2.841ms [vs orig 2.51ms] (+13.18%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v2/nt_bn128_col_bk64_tg64: avg_gpu=2.211ms | avg_cpu=2.404ms [vs orig 2.51ms] (-4.23%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v2/nt_bn128_col_vec4_bk128_tg128: avg_gpu=2.06ms | avg_cpu=2.222ms [vs orig 2.51ms] (-11.48%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v2/nt_bn128_col_vec4_bk64_tg128: avg_gpu=2.312ms | avg_cpu=2.536ms [vs orig 2.51ms] (+1.02%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v2/nt_bn64_col: avg_gpu=2.038ms | avg_cpu=2.228ms [vs orig 2.51ms] (-11.24%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v2/nt_bn64_col_bk128_tg128: avg_gpu=2.282ms | avg_cpu=2.483ms [vs orig 2.51ms] (-1.08%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v2/nt_bn64_col_bk64_tg128: avg_gpu=2.034ms | avg_cpu=2.228ms [vs orig 2.51ms] (-11.23%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v2/nt_bn64_col_bk64_tg64: avg_gpu=2.128ms | avg_cpu=2.326ms [vs orig 2.51ms] (-7.32%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v2/nt_bn64_col_vec4_bk128_tg128: avg_gpu=2.053ms | avg_cpu=2.226ms [vs orig 2.51ms] (-11.33%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v2/nt_bn64_col_vec4_bk64_tg128: avg_gpu=2.066ms | avg_cpu=2.253ms [vs orig 2.51ms] (-10.23%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v3/nt_bn128_col_vec4_sgbr_bk64_tg128: avg_gpu=2.091ms | avg_cpu=2.289ms [vs orig 2.51ms] (-8.81%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v3/nt_bn128_col_vec4_tgread_bk128_tg128: avg_gpu=2.058ms | avg_cpu=2.257ms [vs orig 2.51ms] (-10.09%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v3/nt_bn128_col_vec4_tgread_bk64_tg128: avg_gpu=2.076ms | avg_cpu=2.265ms [vs orig 2.51ms] (-9.74%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v3/nt_bn64_col_vec4_tgread_bk128_tg128: avg_gpu=2.049ms | avg_cpu=2.238ms [vs orig 2.51ms] (-10.83%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v4/nt_bn256_col_vec4_tgread_bk128_tg128: avg_gpu=2.282ms | avg_cpu=2.437ms [vs orig 2.51ms] (-2.92%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v4/nt_bn256_col_vec4_tgread_bk64_tg128: avg_gpu=2.357ms | avg_cpu=2.644ms [vs orig 2.51ms] (+5.32%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v4/nt_tiny_bn32_tg64: avg_gpu=2.024ms | avg_cpu=2.201ms [vs orig 2.51ms] (-12.33%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v4/nt_tiny_bn64_tg64: avg_gpu=2.047ms | avg_cpu=2.248ms [vs orig 2.51ms] (-10.42%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v6/nt_bn128_col_vec4_tgread_vA_bk128_tg128: avg_gpu=2.44ms | avg_cpu=2.638ms [vs orig 2.51ms] (+5.08%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v6/nt_bn128_col_vec4_tgread_vA_bk64_tg128: avg_gpu=2.006ms | avg_cpu=2.189ms [vs orig 2.51ms] (-12.79%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v6/nt_bn160_col_vec4_tgread_vA_bk128_tg128: avg_gpu=2.245ms | avg_cpu=2.432ms [vs orig 2.51ms] (-3.09%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v6/nt_bn160_col_vec4_tgread_vA_bk64_tg128: avg_gpu=2.652ms | avg_cpu=2.938ms [vs orig 2.51ms] (+17.07%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v6/nt_bn192_col_vec4_tgread_vA_bk128_tg128: avg_gpu=2.09ms | avg_cpu=2.269ms [vs orig 2.51ms] (-9.59%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v6/nt_bn192_col_vec4_tgread_vA_bk64_tg128: avg_gpu=2.158ms | avg_cpu=2.339ms [vs orig 2.51ms] (-6.82%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v6/nt_bn224_col_vec4_tgread_vA_bk128_tg128: avg_gpu=2.017ms | avg_cpu=2.204ms [vs orig 2.51ms] (-12.21%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v6/nt_bn224_col_vec4_tgread_vA_bk64_tg128: avg_gpu=2.057ms | avg_cpu=2.248ms [vs orig 2.51ms] (-10.45%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v6/nt_bn256_col_vec4_tgread_vA_bk128_tg128: avg_gpu=2.131ms | avg_cpu=2.35ms [vs orig 2.51ms] (-6.39%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v6/nt_bn256_col_vec4_tgread_vA_bk64_tg128: avg_gpu=2.04ms | avg_cpu=2.228ms [vs orig 2.51ms] (-11.22%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v6/nt_bn256_col_vec4_tgread_vA_bk64_tg256: avg_gpu=2.752ms | avg_cpu=2.96ms [vs orig 2.51ms] (+17.92%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v6/nt_bn64_col_vec4_tgread_vA_bk128_tg128 [best-gpu] [best-cpu]: avg_gpu=2.005ms | avg_cpu=2.183ms [vs orig 2.51ms] (-13.02%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v6/nt_bn64_col_vec4_tgread_vA_bk64_tg128: avg_gpu=2.236ms | avg_cpu=2.425ms [vs orig 2.51ms] (-3.38%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v7/nt_adaptive_bn128_tg128: avg_gpu=2.19ms | avg_cpu=2.376ms [vs orig 2.51ms] (-5.33%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v7/nt_bn128_col_vec4_triplebuf_bk64_tg128: avg_gpu=2.088ms | avg_cpu=2.265ms [vs orig 2.51ms] (-9.76%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v7/nt_bn128_col_vec4_worksteal_bk128_tg128: avg_gpu=2.321ms | avg_cpu=2.512ms [vs orig 2.51ms] (+0.10%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v7/nt_bn128_col_vec4_worksteal_chunk4_bk128_tg128: avg_gpu=2.568ms | avg_cpu=2.867ms [vs orig 2.51ms] (+14.21%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v7/nt_bn64_col_vec4_transformB_bk64_tg128: avg_gpu=2.829ms | avg_cpu=3.015ms [vs orig 2.51ms] (+20.11%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - m1_optimized_v7/nt_bn64_col_vec4_worksteal_chunk4_bk128_tg128: avg_gpu=2.464ms | avg_cpu=2.668ms [vs orig 2.51ms] (+6.31%) maxAbs=1.5606e-02 maxRel=4.8769e-04
   - mps/baseline [baseline]: avg_gpu=2.346ms | avg_cpu=2.508ms [vs orig 2.51ms] (-0.06%) maxAbs=1.5606e-02 maxRel=4.8769e-04
Spec op=matmul_alpha_beta | batch=14 | m=1 | n=64 | k=95 | tA=0 | tB=0:
   - mlx/baseline_nn [baseline] [best-cpu]: avg_gpu=0.007ms | avg_cpu=0.094ms [vs orig 0.25ms] (-62.33%) maxAbs=3.7918e-03 maxRel=4.5909e-04
   - mlx/bk64_nn [best-gpu]: avg_gpu=0.006ms | avg_cpu=0.104ms [vs orig 0.25ms] (-58.50%) maxAbs=3.7918e-03 maxRel=4.5909e-04
   - mlx/m1_fast_nn: avg_gpu=0.011ms | avg_cpu=0.159ms [vs orig 0.25ms] (-36.26%) maxAbs=3.7918e-03 maxRel=4.5909e-04
   - mlx/nn_fast: avg_gpu=0.059ms | avg_cpu=0.281ms [vs orig 0.25ms] (+12.55%) maxAbs=3.7918e-03 maxRel=4.5909e-04
   - mlx/nn_wide: avg_gpu=0.009ms | avg_cpu=0.139ms [vs orig 0.25ms] (-44.34%) maxAbs=3.7918e-03 maxRel=4.5909e-04
   - mps/baseline: avg_gpu=0.045ms | avg_cpu=0.185ms [vs orig 0.25ms] (-25.86%) maxAbs=3.7918e-03 maxRel=4.5909e-04
Spec op=matmul_alpha_beta | batch=14 | m=1 | n=64 | k=16 | tA=0 | tB=0:
   - mlx/baseline_nn [baseline] [best-gpu] [best-cpu]: avg_gpu=0.004ms | avg_cpu=0.095ms [vs orig 0.249ms] (-61.98%) maxAbs=9.7466e-04 maxRel=4.7078e-04
   - mlx/bk64_nn: avg_gpu=0.005ms | avg_cpu=0.123ms [vs orig 0.249ms] (-50.72%) maxAbs=9.7466e-04 maxRel=4.7078e-04
   - mlx/m1_fast_nn: avg_gpu=0.007ms | avg_cpu=0.153ms [vs orig 0.249ms] (-38.64%) maxAbs=9.7466e-04 maxRel=4.7078e-04
   - mlx/nn_fast: avg_gpu=0.005ms | avg_cpu=0.122ms [vs orig 0.249ms] (-50.90%) maxAbs=9.7466e-04 maxRel=4.7078e-04
   - mlx/nn_wide: avg_gpu=0.006ms | avg_cpu=0.122ms [vs orig 0.249ms] (-50.93%) maxAbs=9.7466e-04 maxRel=4.7078e-04
   - mps/baseline: avg_gpu=0.004ms | avg_cpu=0.148ms [vs orig 0.249ms] (-40.43%) maxAbs=9.7466e-04 maxRel=4.7078e-04
Spec op=matmul_alpha_beta | batch=14 | m=1 | n=51 | k=64 | tA=0 | tB=1:
   - mlx/baseline [baseline]: avg_gpu=0.006ms | avg_cpu=0.125ms [vs orig 0.231ms] (-45.79%) maxAbs=2.6875e-03 maxRel=4.5867e-04
   - mlx/bk64_nt: avg_gpu=0.029ms | avg_cpu=0.161ms [vs orig 0.231ms] (-30.18%) maxAbs=2.6875e-03 maxRel=4.5867e-04
   - mlx/m1_fast_nt: avg_gpu=0.007ms | avg_cpu=0.145ms [vs orig 0.231ms] (-37.37%) maxAbs=2.6875e-03 maxRel=4.5867e-04
   - mps/baseline [best-gpu] [best-cpu]: avg_gpu=0.003ms | avg_cpu=0.11ms [vs orig 0.231ms] (-52.29%) maxAbs=2.6875e-03 maxRel=4.5867e-04
Spec op=matmul_alpha_beta | batch=14 | m=1 | n=64 | k=88 | tA=0 | tB=0:
   - mlx/baseline_nn [baseline]: avg_gpu=0.009ms | avg_cpu=0.124ms [vs orig 0.219ms] (-43.32%) maxAbs=3.3283e-03 maxRel=4.6677e-04
   - mlx/bk64_nn: avg_gpu=0.006ms | avg_cpu=0.109ms [vs orig 0.219ms] (-50.20%) maxAbs=3.3283e-03 maxRel=4.6677e-04
   - mlx/m1_fast_nn: avg_gpu=0.011ms | avg_cpu=0.161ms [vs orig 0.219ms] (-26.31%) maxAbs=3.3283e-03 maxRel=4.6677e-04
   - mlx/nn_fast: avg_gpu=0.007ms | avg_cpu=0.114ms [vs orig 0.219ms] (-47.80%) maxAbs=3.3283e-03 maxRel=4.6677e-04
   - mlx/nn_wide: avg_gpu=0.009ms | avg_cpu=0.128ms [vs orig 0.219ms] (-41.75%) maxAbs=3.3283e-03 maxRel=4.6677e-04
   - mps/baseline [best-gpu] [best-cpu]: avg_gpu=0.004ms | avg_cpu=0.092ms [vs orig 0.219ms] (-57.84%) maxAbs=3.3283e-03 maxRel=4.6677e-04
Spec op=matmul_alpha_beta | batch=14 | m=1 | n=83 | k=64 | tA=0 | tB=1:
   - mlx/baseline [baseline]: avg_gpu=0.007ms | avg_cpu=0.12ms [vs orig 0.216ms] (-44.58%) maxAbs=2.2125e-03 maxRel=4.5933e-04
   - mlx/bk64_nt: avg_gpu=0.006ms | avg_cpu=0.114ms [vs orig 0.216ms] (-47.34%) maxAbs=2.2125e-03 maxRel=4.5933e-04
   - mlx/m1_fast_nt: avg_gpu=0.006ms | avg_cpu=0.126ms [vs orig 0.216ms] (-41.62%) maxAbs=2.2125e-03 maxRel=4.5933e-04
   - mps/baseline [best-gpu] [best-cpu]: avg_gpu=0.003ms | avg_cpu=0.092ms [vs orig 0.216ms] (-57.35%) maxAbs=2.2125e-03 maxRel=4.5933e-04
Spec op=matmul_alpha_beta | batch=14 | m=1 | n=90 | k=64 | tA=0 | tB=1:
   - mlx/baseline [baseline]: avg_gpu=0.007ms | avg_cpu=0.119ms [vs orig 0.208ms] (-43.02%) maxAbs=3.8767e-03 maxRel=4.7921e-04
   - mlx/bk64_nt: avg_gpu=0.006ms | avg_cpu=0.114ms [vs orig 0.208ms] (-45.30%) maxAbs=3.8767e-03 maxRel=4.7921e-04
   - mlx/m1_fast_nt: avg_gpu=0.008ms | avg_cpu=0.15ms [vs orig 0.208ms] (-28.03%) maxAbs=3.8767e-03 maxRel=4.7921e-04
   - mps/baseline [best-gpu] [best-cpu]: avg_gpu=0.003ms | avg_cpu=0.096ms [vs orig 0.208ms] (-53.87%) maxAbs=3.8767e-03 maxRel=4.7921e-04
Spec op=matmul_alpha_beta | batch=14 | m=1 | n=76 | k=64 | tA=0 | tB=1:
   - mlx/baseline [baseline] [best-cpu]: avg_gpu=0.006ms | avg_cpu=0.108ms [vs orig 0.208ms] (-47.88%) maxAbs=2.2125e-03 maxRel=4.5971e-04
   - mlx/bk64_nt: avg_gpu=0.006ms | avg_cpu=0.113ms [vs orig 0.208ms] (-45.78%) maxAbs=2.2125e-03 maxRel=4.5971e-04
   - mlx/m1_fast_nt: avg_gpu=0.006ms | avg_cpu=0.152ms [vs orig 0.208ms] (-27.10%) maxAbs=2.2125e-03 maxRel=4.5971e-04
   - mps/baseline [best-gpu]: avg_gpu=0.004ms | avg_cpu=0.161ms [vs orig 0.208ms] (-22.48%) maxAbs=2.2125e-03 maxRel=4.5971e-04
Spec op=matmul_alpha_beta | batch=14 | m=1 | n=64 | k=83 | tA=0 | tB=0:
   - mlx/baseline_nn [baseline]: avg_gpu=0.007ms | avg_cpu=0.118ms [vs orig 0.187ms] (-36.77%) maxAbs=3.7842e-03 maxRel=4.6709e-04
   - mlx/bk64_nn: avg_gpu=0.006ms | avg_cpu=0.129ms [vs orig 0.187ms] (-30.99%) maxAbs=3.7842e-03 maxRel=4.6709e-04
   - mlx/m1_fast_nn: avg_gpu=0.01ms | avg_cpu=0.158ms [vs orig 0.187ms] (-15.30%) maxAbs=3.7842e-03 maxRel=4.6709e-04
   - mlx/nn_fast: avg_gpu=0.009ms | avg_cpu=0.123ms [vs orig 0.187ms] (-34.01%) maxAbs=3.7842e-03 maxRel=4.6709e-04
   - mlx/nn_wide: avg_gpu=0.009ms | avg_cpu=0.133ms [vs orig 0.187ms] (-29.00%) maxAbs=3.7842e-03 maxRel=4.6709e-04
   - mps/baseline [best-gpu] [best-cpu]: avg_gpu=0.004ms | avg_cpu=0.117ms [vs orig 0.187ms] (-37.36%) maxAbs=3.7842e-03 maxRel=4.6709e-04
Spec op=matmul_alpha_beta | batch=14 | m=1 | n=64 | k=69 | tA=0 | tB=0:
   - mlx/baseline_nn [baseline]: avg_gpu=0.008ms | avg_cpu=0.183ms [vs orig 0.176ms] (+4.10%) maxAbs=2.2001e-03 maxRel=4.7458e-04
   - mlx/bk64_nn: avg_gpu=0.006ms | avg_cpu=0.153ms [vs orig 0.176ms] (-13.30%) maxAbs=2.2001e-03 maxRel=4.7458e-04
   - mlx/m1_fast_nn: avg_gpu=0.012ms | avg_cpu=0.164ms [vs orig 0.176ms] (-6.82%) maxAbs=2.2001e-03 maxRel=4.7458e-04
   - mlx/nn_fast: avg_gpu=0.007ms | avg_cpu=0.118ms [vs orig 0.176ms] (-32.76%) maxAbs=2.2001e-03 maxRel=4.7458e-04
   - mlx/nn_wide: avg_gpu=0.011ms | avg_cpu=0.159ms [vs orig 0.176ms] (-9.73%) maxAbs=2.2001e-03 maxRel=4.7458e-04
   - mps/baseline [best-gpu] [best-cpu]: avg_gpu=0.004ms | avg_cpu=0.092ms [vs orig 0.176ms] (-47.85%) maxAbs=2.2001e-03 maxRel=4.7458e-04
Spec op=matmul_alpha_beta | batch=14 | m=1 | n=80 | k=64 | tA=0 | tB=1:
   - mlx/baseline [baseline]: avg_gpu=0.008ms | avg_cpu=0.192ms [vs orig 0.176ms] (+9.14%) maxAbs=2.2125e-03 maxRel=4.5933e-04
   - mlx/bk64_nt: avg_gpu=0.01ms | avg_cpu=0.192ms [vs orig 0.176ms] (+8.90%) maxAbs=2.2125e-03 maxRel=4.5933e-04
   - mlx/m1_fast_nt: avg_gpu=0.007ms | avg_cpu=0.144ms [vs orig 0.176ms] (-18.45%) maxAbs=2.2125e-03 maxRel=4.5933e-04
   - mps/baseline [best-gpu] [best-cpu]: avg_gpu=0.003ms | avg_cpu=0.092ms [vs orig 0.176ms] (-47.67%) maxAbs=2.2125e-03 maxRel=4.5933e-04
Spec op=matmul_alpha_beta | batch=14 | m=1 | n=67 | k=64 | tA=0 | tB=1:
   - mlx/baseline [baseline]: avg_gpu=0.006ms | avg_cpu=0.117ms [vs orig 0.172ms] (-31.96%) maxAbs=2.2125e-03 maxRel=4.7609e-04
   - mlx/bk64_nt: avg_gpu=0.006ms | avg_cpu=0.109ms [vs orig 0.172ms] (-36.90%) maxAbs=2.2125e-03 maxRel=4.7609e-04
   - mlx/m1_fast_nt: avg_gpu=0.005ms | avg_cpu=0.125ms [vs orig 0.172ms] (-27.08%) maxAbs=2.2125e-03 maxRel=4.7609e-04
   - mps/baseline [best-gpu] [best-cpu]: avg_gpu=0.003ms | avg_cpu=0.093ms [vs orig 0.172ms] (-45.96%) maxAbs=2.2125e-03 maxRel=4.7609e-04
Spec op=matmul_alpha_beta | batch=14 | m=1 | n=64 | k=67 | tA=0 | tB=0:
   - mlx/baseline_nn [baseline]: avg_gpu=0.007ms | avg_cpu=0.122ms [vs orig 0.172ms] (-29.18%) maxAbs=1.9507e-03 maxRel=4.6554e-04
   - mlx/bk64_nn: avg_gpu=0.006ms | avg_cpu=0.147ms [vs orig 0.172ms] (-14.38%) maxAbs=1.9507e-03 maxRel=4.6554e-04
   - mlx/m1_fast_nn: avg_gpu=0.011ms | avg_cpu=0.162ms [vs orig 0.172ms] (-5.54%) maxAbs=1.9507e-03 maxRel=4.6554e-04
   - mlx/nn_fast: avg_gpu=0.006ms | avg_cpu=0.105ms [vs orig 0.172ms] (-38.92%) maxAbs=1.9507e-03 maxRel=4.6554e-04
   - mlx/nn_wide: avg_gpu=0.012ms | avg_cpu=0.162ms [vs orig 0.172ms] (-5.61%) maxAbs=1.9507e-03 maxRel=4.6554e-04
   - mps/baseline [best-gpu] [best-cpu]: avg_gpu=0.004ms | avg_cpu=0.091ms [vs orig 0.172ms] (-47.30%) maxAbs=1.9507e-03 maxRel=4.6554e-04
Spec op=matmul_alpha_beta | batch=14 | m=1 | n=73 | k=64 | tA=0 | tB=1:
   - mlx/baseline [baseline]: avg_gpu=0.006ms | avg_cpu=0.118ms [vs orig 0.172ms] (-31.41%) maxAbs=2.2125e-03 maxRel=4.5971e-04
   - mlx/bk64_nt: avg_gpu=0.006ms | avg_cpu=0.115ms [vs orig 0.172ms] (-33.27%) maxAbs=2.2125e-03 maxRel=4.5971e-04
   - mlx/m1_fast_nt: avg_gpu=0.007ms | avg_cpu=0.147ms [vs orig 0.172ms] (-14.71%) maxAbs=2.2125e-03 maxRel=4.5971e-04
   - mps/baseline [best-gpu] [best-cpu]: avg_gpu=0.004ms | avg_cpu=0.107ms [vs orig 0.172ms] (-37.80%) maxAbs=2.2125e-03 maxRel=4.5971e-04
Spec op=matmul_alpha_beta | batch=14 | m=1 | n=64 | k=68 | tA=0 | tB=0:
   - mlx/baseline_nn [baseline]: avg_gpu=0.006ms | avg_cpu=0.114ms [vs orig 0.169ms] (-32.26%) maxAbs=1.9526e-03 maxRel=4.6442e-04
   - mlx/bk64_nn: avg_gpu=0.006ms | avg_cpu=0.121ms [vs orig 0.169ms] (-28.43%) maxAbs=1.9526e-03 maxRel=4.6442e-04
   - mlx/m1_fast_nn: avg_gpu=0.011ms | avg_cpu=0.166ms [vs orig 0.169ms] (-1.52%) maxAbs=1.9526e-03 maxRel=4.6442e-04
   - mlx/nn_fast: avg_gpu=0.013ms | avg_cpu=0.202ms [vs orig 0.169ms] (+19.76%) maxAbs=1.9526e-03 maxRel=4.6442e-04
   - mlx/nn_wide: avg_gpu=0.011ms | avg_cpu=0.158ms [vs orig 0.169ms] (-6.51%) maxAbs=1.9526e-03 maxRel=4.6442e-04
   - mps/baseline [best-gpu] [best-cpu]: avg_gpu=0.004ms | avg_cpu=0.105ms [vs orig 0.169ms] (-37.70%) maxAbs=1.9526e-03 maxRel=4.6442e-04
Spec op=matmul_alpha_beta | batch=14 | m=1 | n=64 | k=84 | tA=0 | tB=0:
   - mlx/baseline_nn [baseline]: avg_gpu=0.008ms | avg_cpu=0.12ms [vs orig 0.169ms] (-28.85%) maxAbs=3.4161e-03 maxRel=4.6089e-04
   - mlx/bk64_nn: avg_gpu=0.006ms | avg_cpu=0.118ms [vs orig 0.169ms] (-30.15%) maxAbs=3.4161e-03 maxRel=4.6089e-04
   - mlx/m1_fast_nn: avg_gpu=0.011ms | avg_cpu=0.163ms [vs orig 0.169ms] (-3.40%) maxAbs=3.4161e-03 maxRel=4.6089e-04
   - mlx/nn_fast: avg_gpu=0.007ms | avg_cpu=0.119ms [vs orig 0.169ms] (-29.63%) maxAbs=3.4161e-03 maxRel=4.6089e-04
   - mlx/nn_wide: avg_gpu=0.011ms | avg_cpu=0.168ms [vs orig 0.169ms] (-0.48%) maxAbs=3.4161e-03 maxRel=4.6089e-04
   - mps/baseline [best-gpu] [best-cpu]: avg_gpu=0.004ms | avg_cpu=0.112ms [vs orig 0.169ms] (-33.56%) maxAbs=3.4161e-03 maxRel=4.6089e-04

Variant failures:
 - mlx/baseline: matmul_alpha_beta skipped 8 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - mlx/baseline_nn: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 7 (unsupported)
 - mlx/bk64_nn: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 7 (unsupported)
 - mlx/bk64_nt: matmul_alpha_beta skipped 8 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - mlx/bk64_tn: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - mlx/bk64_tt: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - mlx/m1_fast_nn: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 7 (unsupported)
 - mlx/m1_fast_nt: matmul_alpha_beta skipped 8 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - mlx/m1_fast_tn: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - mlx/m1_fast_tt: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - mlx/nn_fast: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 7 (unsupported)
 - mlx/nn_wide: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 7 (unsupported)
 - m1_optimized_v2/nt_bn128_col: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v2/nt_bn128_col_bk128_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v2/nt_bn128_col_bk64_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v2/nt_bn128_col_bk64_tg64: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v2/nt_bn128_col_vec4_bk128_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v2/nt_bn128_col_vec4_bk64_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v2/nt_bn64_col: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v2/nt_bn64_col_bk128_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v2/nt_bn64_col_bk64_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v2/nt_bn64_col_bk64_tg64: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v2/nt_bn64_col_vec4_bk128_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v2/nt_bn64_col_vec4_bk64_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v3/nt_bn128_col_vec4_sgbr_bk64_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v3/nt_bn128_col_vec4_tgread_bk128_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v3/nt_bn128_col_vec4_tgread_bk64_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v3/nt_bn64_col_vec4_tgread_bk128_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v4/nt_bn16_largek_smalln_tg256: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v4/nt_bn256_col_vec4_tgread_bk128_tg128: matmul skipped 2 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v4/nt_bn256_col_vec4_tgread_bk64_tg128: matmul skipped 2 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v4/nt_bn4_largek_smalln_tg256: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v4/nt_bn8_largek_smalln_tg128: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v4/nt_bn8_largek_smalln_tg256: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v4/nt_tiny_bn32_tg64: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v4/nt_tiny_bn64_tg64: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v5/nt_bn16_largek_smalln_tg256: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v5/nt_bn16_largek_smalln_tg512: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v5/nt_bn4_largek_smalln_tg256: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v5/nt_bn4_largek_smalln_tg512: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v5/nt_bn8_largek_smalln_tg256: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v5/nt_bn8_largek_smalln_tg512: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v5/nt_ultra_tiny_single_tg32: matmul skipped 3 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn128_col_vec4_tgread_vA_bk128_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn128_col_vec4_tgread_vA_bk64_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn160_col_vec4_tgread_vA_bk128_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn160_col_vec4_tgread_vA_bk64_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn16_largek_smalln_kpar_bk128_tg128: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn16_largek_smalln_kpar_bk256_tg256: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn16_largek_smalln_kpar_bk64_tg128: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn192_col_vec4_tgread_vA_bk128_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn192_col_vec4_tgread_vA_bk64_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn224_col_vec4_tgread_vA_bk128_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn224_col_vec4_tgread_vA_bk64_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn256_col_vec4_tgread_vA_bk128_tg128: matmul skipped 2 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn256_col_vec4_tgread_vA_bk64_tg128: matmul skipped 2 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn256_col_vec4_tgread_vA_bk64_tg256: matmul skipped 2 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn32_largek_smalln_kpar_bk128_tg128: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn32_largek_smalln_kpar_bk256_tg256: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn32_largek_smalln_kpar_bk64_tg128: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn4_largek_smalln_kpar_bk128_tg128: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn4_largek_smalln_kpar_bk256_tg256: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn64_col_vec4_tgread_vA_bk128_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn64_col_vec4_tgread_vA_bk64_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn64_largek_smalln_kpar_bk128_tg128: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn64_largek_smalln_kpar_bk256_tg256: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn8_largek_smalln_kpar_bk128_tg128: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn8_largek_smalln_kpar_bk256_tg256: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v6/nt_bn8_largek_smalln_kpar_bk64_tg128: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v7/nt_adaptive_bn128_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v7/nt_bn128_col_vec4_triplebuf_bk64_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v7/nt_bn128_col_vec4_worksteal_bk128_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v7/nt_bn128_col_vec4_worksteal_chunk4_bk128_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v7/nt_bn16_largek_smalln_kpar_vec4_both_bk256_tg256: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v7/nt_bn16_smalln_simdgroupmm_tg256: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v7/nt_bn256_fused_bias_gelu_tg256: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v7/nt_bn256_fused_bias_relu_tg256: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v7/nt_bn64_col_vec4_transformB_bk64_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v7/nt_bn64_col_vec4_worksteal_chunk4_bk128_tg128: matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v7/nt_bn8_largek_smalln_hybrid_kpar_bk256_tg256: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - m1_optimized_v7/nt_bn8_largek_smalln_kpar_vec4_both_bk256_tg256: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - gemv/baseline: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported)
 - gemv/smalln: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported), matmul_bias_add skipped 1 (unsupported)
 - gemm_tiled/baseline: matmul skipped 4 (unsupported), matmul_alpha_beta skipped 15 (unsupported)
   (use --verbose-failures to list every skipped spec)
