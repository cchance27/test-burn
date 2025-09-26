# Tokenizer Performance: A Summary of Benchmark-Driven Optimizations

This document summarizes the successful, iterative process of optimizing the BPE tokenizer. Through a series of changes and benchmarks, we significantly improved performance and memory efficiency.

## 1. The Optimization Journey

We implemented several layers of optimization, each targeting a specific bottleneck:

1.  **ID-Based BPE Logic:** The initial and most impactful change was to replace the string-based BPE merge logic with an integer-based (`u32`) algorithm. This drastically reduced memory allocations and improved cache efficiency.

2.  **Allocation-Free Lookups:** We introduced a `FxHashMap<char, u32>` as a cache for single-character tokens. This eliminated string allocations that were happening for every character lookup at the start of the BPE process.

3.  **Streaming Piece Processor:** The intermediate `Vec<String>` that held text pieces from the regex split was eliminated for serial paths. A closure-based design now streams `&str` slices directly, avoiding this major allocation.

4.  **Micro-Optimizations:** Other small but important changes included replacing the byte-lookup `FxHashMap` with a simple array for O(1) access and pre-allocating intermediate buffers to reduce reallocations.

5.  **Algorithmic Investigation:** We experimented with a theoretically faster O(N log N) BPE merge algorithm using a priority queue. However, benchmarks proved it was slower for our workload due to high constant-factor overhead. We correctly reverted to the simpler and practically faster O(N²) ID-based algorithm.

## 2. Final Outcome and Key Findings

The final benchmarks after all optimizations revealed a clear winner:

- **`encode_simd` is the Fastest Overall:** The fully optimized, single-threaded `encode_simd` function consistently outperformed all other variants—including the parallel ones—across all tested text lengths.

- **Overhead vs. Optimization:** The core `bpe_encode_ids` function became so fast that the overhead of parallelization (using `rayon`) was greater than the benefit of distributing the work. This is a testament to how effective the serial optimizations were.

- **Final Implementation:** Based on these conclusive findings, the default `encode` method for the tokenizer has been updated to use `encode_simd`, ensuring the best performance by default.

The tokenizer is now in a highly optimized state, validated by a rigorous, benchmark-driven process. The final implementation is significantly faster and more memory-efficient than the original.