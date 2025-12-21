# 5.2 Kernel Tuning Guide

This guide explains how to tune the dispatcher heuristics for optimal performance across different hardware configurations and model architectures.

## Overview

The 5.2 implementation includes several tunable parameters that affect kernel selection:

1. **Softmax vec/block crossover threshold** (currently 1024)
2. **Small-N GEMV N thresholds** (currently N=1,2,4,8,16)
3. **SIMD GEMM M/N thresholds** (currently M>=64, N>=16)

## Tuning Process

### Benchmark Shape Semantics (Important)

- Softmax operates across the keys dimension (`seq_k`) for each query position. The dispatcher selects `vec` vs `block` based on `seq_k`.
- Use tensors shaped as `[rows_total, seq_k]` for 2D, or `[batch, seq_q, seq_k]` for 3D. Vary `seq_k` on the last dimension.
- Labels should make axis intent explicit, e.g. `rows{R}_seqk{K}` or `batch{B}_seqq{Q}_seqk{K}`. Avoid varying `rows_total` when analyzing softmax variant crossover.

### 1. Run Comprehensive Benchmarks

```bash
# Run all benchmarks to generate performance data
cargo build --release --benches
cargo bench --bench softmax_dispatcher_bench
cargo bench --bench direct_kernel_bench

# Collect results into CSV files for analysis
python benches/tuning_kernels/collect_criterion_csv.py --criterion-dir target/criterion --out benches/benchmark_results.csv
```

### 2. Analyze Results

Use the provided analysis script to find optimal thresholds:

```bash
python benches/tuning_kernels/analyze_crossover_points.py benches/benchmark_results.csv --output-dir benches/tuning_analysis
```

The script will output:
- Optimal softmax crossover points
- Small-N GEMV threshold recommendations
- Performance visualizations

### 3. Update Thresholds

Based on the analysis, update the dispatcher constants:

#### Softmax Threshold
```rust
// In crates/metallic/src/kernels/softmax_dispatcher/dispatcher.rs
const SOFTMAX_VEC_BLOCK_THRESHOLD: usize = 1280; // Tuned from analysis
```

## Hardware-Specific Tuning

Different GPU architectures may have different optimal thresholds:

### M1/M2 Series
- Generally lower memory bandwidth
- May benefit from more conservative thresholds
- Vec-softmax may be preferred for longer sequences

### M3 Series
- Higher memory bandwidth and compute capability
- Can handle larger block-softmax segments
- May benefit from more aggressive SIMD usage

## Model-Specific Considerations

### Small Models (7B-13B parameters)
- Often use smaller batch sizes and sequence lengths
- May benefit from lower Small-N thresholds
- Vec-softmax often sufficient

### Large Models (30B-70B+ parameters)
- Use longer sequence lengths
- May benefit from block-softmax for attention computation
- Higher SIMD thresholds may be beneficial

## Environment Variables for Testing

Use these environment variables to test different thresholds without recompilation:

```bash
# Softmax variant selection
export METALLIC_SOFTMAX_VARIANT=auto  # or vec, block
# Note: `SOFTMAX_BACKEND_VAR` in code is the typed wrapper for this env var.

# Matmul backend selection
export METALLIC_MATMUL_BACKEND=auto  # or mlx, gemv

```

## Validation Process

1. **Run benchmarks** with different threshold values
2. **Analyze performance** using the provided analysis script
3. **Validate correctness** with integration tests
4. **Check memory usage** doesn't regress significantly
5. **Update constants** based on empirical results

## Expected Performance Improvements

Based on the 5.2 implementation (post CSV/axis fixes):

- **Small-N GEMV**: Gains expected for N≤16; validate on target hardware
- **Vec-softmax**: Preferred for `seq_k ≤ 1024` after tuning
- **Block-softmax**: Preferred for `seq_k > 1024`; requires dynamic tiling
- **Causal masking**: Some paths may still fallback; validate and tune
- **Overall SDPA**: Improvements depend on model/hardware; re-benchmark post-fixes

## Monitoring and Alerting

Set up monitoring to detect performance regressions:

1. **CI benchmarks** should run on each PR
2. **Performance gates** should prevent regressions >5%
3. **Hardware-specific benchmarks** for different GPU families
4. **Model-specific benchmarks** for different architectures

## Troubleshooting

### Common Issues

1. **Performance regression**: Check if new thresholds are appropriate for test hardware
2. **Memory increase**: Verify block-softmax segment sizes aren't too large
3. **Correctness failures**: Ensure causal masking logic is correct
4. **Compilation errors**: Check Metal shader compatibility

### Debug Mode

Enable debug logging to see which kernels are selected:

```bash
export METALLIC_LOG_LEVEL=debug
export METALLIC_GPU_PROFILER=true
```

This will output detailed information about kernel selection decisions and performance characteristics.

## CSV Collection and Throughput

- Use `benches/tuning_kernels/collect_criterion_csv.py` to aggregate results.
- Throughput is computed per benchmark based on label semantics:
  - Softmax: `elements = rows_total * seq_k * 3` (read+max+exp+sum+div)
  - Matmul: `elements = M * K * N * 2` (multiply-adds)
- Ensure labels follow the conventions so the collector can infer axis values.
