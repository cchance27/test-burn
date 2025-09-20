# Testing some Burn-rs

I implemented a basic sdpa to try to match pytorch, but noticed benchmarks show that pytorch is a good bit faster, not sure why yet. 

```bash
cargo run --release

Running Metal Opt (MPS) implementation (causal)...
Metal Opt (MPS) time for 100 iterations: 496.04475ms
Metal Opt (MPS) time for 100 iterations: 479.962542ms
Metal Opt (MPS) time for 100 iterations: 480.440708ms
Metal Opt (MPS) time for 100 iterations: 481.387ms
Metal Opt (MPS) time for 100 iterations: 481.231459ms

Running Metal Opt (MPS) implementation (non-causal)...
Metal Opt (MPS) time for 100 iterations: 480.970625ms
Metal Opt (MPS) time for 100 iterations: 480.19025ms
Metal Opt (MPS) time for 100 iterations: 481.265625ms
Metal Opt (MPS) time for 100 iterations: 480.163833ms
Metal Opt (MPS) time for 100 iterations: 481.812709ms

Running Metal (MPS) implementation...
Metal (MPS) time for 100 iterations: 1.338323417s
Metal (MPS) time for 100 iterations: 1.347275708s
Metal (MPS) time for 100 iterations: 1.347281875s
Metal (MPS) time for 100 iterations: 1.341438458s
Metal (MPS) time for 100 iterations: 1.349358625s

Running Burn implementation...
Burn time for 100 iterations: 1.13353025s
Burn time for 100 iterations: 1.046201917s
Burn time for 100 iterations: 1.061525s
Burn time for 100 iterations: 1.070377917s
Burn time for 100 iterations: 1.0730905s

Running SDPA Custom Benchmarks...
=== SDPA Metal Performance Benchmark ===

Benchmarking SDPA with batch=4, seq_q=128, seq_k=128, dim=64, iterations=50, causal=false
Warming up...
Running benchmark...
Completed iteration 0
Completed iteration 10
Completed iteration 20
Completed iteration 30
Completed iteration 40
Time for 50 iterations: 24.38ms
Average time per iteration: 487.57µs
Iterations per second: 2050.99

----------------------------------------

Benchmarking SDPA with batch=4, seq_q=512, seq_k=512, dim=64, iterations=20, causal=false
Warming up...
Running benchmark...
Completed iteration 0
Completed iteration 10
Time for 20 iterations: 7.97ms
Average time per iteration: 398.64µs
Iterations per second: 2508.54

----------------------------------------

Benchmarking SDPA with batch=4, seq_q=1024, seq_k=1024, dim=64, iterations=10, causal=false
Warming up...
Running benchmark...
Completed iteration 0
Time for 10 iterations: 8.40ms
Average time per iteration: 840.45µs
Iterations per second: 1189.84

----------------------------------------

Benchmarking SDPA with batch=4, seq_q=512, seq_k=512, dim=64, iterations=20, causal=true
Warming up...
Running benchmark...
Completed iteration 0
Completed iteration 10
Time for 20 iterations: 7.14ms
Average time per iteration: 356.80µs
Iterations per second: 2802.71

----------------------------------------
```