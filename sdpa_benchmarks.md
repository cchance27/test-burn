6.55s baseline 
4.75s caching descriptors
3.59s with per-batch command buffers
2.34s with custom softmax+masking kernel

Notes and follow-ups
  • Tensor allocation in cache:
     • I used a temporary Context to allocate new buffers because Tensor::create_tensor takes Context. Cleaner design
       would allow allocation directly from device or a small allocator. We can adjust Tensor API or add a
       BufferAllocator to the cache later.
  • Softmax wrapper is now unused:
     • We can remove the softmax module or keep it if you want a fallback path.

 Proposed next steps
  • Tests for larger shapes of sdpa
     • Add more tests using larger shapes (e.g., seq_k=1024) to validate reduction logic and performance
  • Tests for Softmax kernel
     • Add tests for various softmax and mask tests to make sure our kernel is good we can use pytorch to generate what results should be and hardcode the result to our test.
  • Optional: MPS batched GEMM investigation
     • If available on your OS, we can replace per-batch GEMM encodes with a single batched GEMM to further reduce
       CPU overhead (though we’re already very good)
  • Optional: micro-tuning
     • Adjust tg_width vs seq_k tiles (e.g., 128 vs 256) and measure
     • Consider prefetching/ldg for value loads if it helps (probably minor on M-series)
