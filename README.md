# Testing some Burn-rs

I implemented a basic sdpa to try to match pytorch, but noticed benchmarks show that pytorch is a good bit faster, not sure why yet. 

```bash
cargo run --release

Finished `release` profile [optimized] target(s) in 42.19s
 Running `target/release/test-burn`
Burn time for 500 iterations: 4.84379375s
Burn time for 500 iterations: 4.777147542s

python3 pytorch/benchmark.py

PyTorch time for 500 iterations: 3.4631659984588623 seconds
PyTorch time for 500 iterations: 3.445702075958252 seconds
```