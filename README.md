# Testing some Burn-rs

I implemented a basic sdpa to try to match pytorch, but noticed benchmarks show that pytorch is a good bit faster, not sure why yet.

## Runtime error alerts

Set the `TEST_BURN_ERROR_LOG` environment variable to enable timestamped error logging. When this variable is set to a file path (or to a truthy value such as `1` to use the default `test-burn-error.log`), runtime errors and warnings are appended to the log. All captured alerts also surface inside the TUI as modal dialogs that can be dismissed with <kbd>Enter</kbd>, <kbd>Esc</kbd>, or <kbd>Space</kbd>.

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

```bash
cargo bench -- --quick --format terse
    Finished `bench` profile [optimized] target(s) in 0.14s
     Running benches/tokenizer_benchmark.rs (target/release/deps/tokenizer_benchmark-7cf6ea83399907f3)
short_encoding/serial   time:   [157.92 ns 158.76 ns 158.98 ns]
                        change: [−3.4419% −2.8372% −2.2289%] (p = 0.07 > 0.05)
                        No change in performance detected.
short_encoding/simd     time:   [155.43 ns 155.91 ns 156.04 ns]
                        change: [+3.3428% +3.5868% +3.8311%] (p = 0.10 > 0.05)
                        No change in performance detected.
short_encoding/parallel time:   [17.937 µs 18.006 µs 18.279 µs]
                        change: [−6.6974% −4.2061% −1.6285%] (p = 0.10 > 0.05)
                        No change in performance detected.
short_encoding/simd_parallel
                        time:   [18.818 µs 19.060 µs 19.121 µs]
                        change: [−2.1947% +0.8144% +3.9622%] (p = 0.81 > 0.05)
                        No change in performance detected.

medium_encoding/serial  time:   [930.75 ns 933.46 ns 934.14 ns]
                        change: [−1.2689% −0.9067% −0.5432%] (p = 0.07 > 0.05)
                        No change in performance detected.
medium_encoding/simd    time:   [927.28 ns 927.95 ns 930.64 ns]
                        change: [−1.6597% −1.2261% −0.7903%] (p = 0.07 > 0.05)
                        No change in performance detected.
medium_encoding/parallel
                        time:   [36.419 µs 36.719 µs 37.919 µs]
                        change: [−3.7501% −0.0976% +3.6813%] (p = 0.87 > 0.05)
                        No change in performance detected.
medium_encoding/simd_parallel
                        time:   [36.170 µs 36.578 µs 38.209 µs]
                        change: [−5.7552% −2.5055% +0.7843%] (p = 0.54 > 0.05)
                        No change in performance detected.

long_encoding/serial    time:   [36.899 µs 36.995 µs 37.377 µs]
                        change: [−1.0104% −0.0102% +0.9973%] (p = 0.86 > 0.05)
                        No change in performance detected.
long_encoding/simd      time:   [37.479 µs 37.541 µs 37.790 µs]
                        change: [−0.6031% +0.2233% +1.0565%] (p = 0.73 > 0.05)
                        No change in performance detected.
long_encoding/parallel  time:   [506.62 µs 508.07 µs 513.90 µs]
                        change: [−1.0084% +0.9716% +3.0026%] (p = 0.61 > 0.05)
                        No change in performance detected.
long_encoding/simd_parallel
                        time:   [487.51 µs 497.89 µs 500.49 µs]
                        change: [+0.1708% +1.8568% +3.5545%] (p = 0.15 > 0.05)
                        No change in performance detected.
```