These are the sizes we found in analyze_jsonl.py with our profiling enabled so that we can perform experimental testing on matmul backends and optimization using direct swift kernel benchmarking to avoid having to test them in the larger metallic crate, once we narrow down improvements, we can look to work them into the proper metallic kernels, to replace the various mlx/gemv/other kernels. 

We want to ensure that we're testing and benching all of them against the originals and our experiments and validating they are correct still.

Matmul backend summary:
  backend=mlx: count=9120 | total=1792.79ms | avg=0.197ms | min=0.107ms | max=1.834ms | p95_ms=0.241ms, p99_ms=0.265ms
  backend=mps: count=4655 | total=1524.37ms | avg=0.327ms | min=0.198ms | max=5.079ms | p95_ms=0.339ms, p99_ms=2.505ms

Matmul op/backend summary:
  matmul_cache @ mps: count=4560 | total=1286.23ms | avg=0.282ms | min=0.198ms | max=5.079ms | p95_ms=0.330ms, p99_ms=0.367ms
  matmul_alpha_beta_cache @ mlx: count=4560 | total=723.39ms | avg=0.159ms | min=0.107ms | max=1.564ms | p95_ms=0.170ms, p99_ms=0.203ms
  matmul_bias_add @ mlx: count=2280 | total=535.99ms | avg=0.235ms | min=0.196ms | max=1.834ms | p95_ms=0.244ms, p99_ms=0.278ms
  matmul @ mlx: count=2280 | total=533.41ms | avg=0.234ms | min=0.188ms | max=1.495ms | p95_ms=0.243ms, p99_ms=0.277ms
  matmul @ mps: count=95 | total=238.14ms | avg=2.507ms | min=2.490ms | max=2.582ms | p95_ms=2.546ms, p99_ms=2.558ms

Matmul shape backend summary:
  op=matmul_cache | batch=1 | m=1 | n=9728 | k=896 | tA=0 | tB=1 | strided_batch=false:
    backend=mps: count=2280 | total=738.37ms | avg=0.324ms | min=0.277ms | max=5.079ms | p95_ms=0.341ms, p99_ms=0.391ms
  op=matmul_cache | batch=1 | m=1 | n=896 | k=4864 | tA=0 | tB=1 | strided_batch=false:
    backend=mps: count=2280 | total=547.86ms | avg=0.240ms | min=0.198ms | max=1.575ms | p95_ms=0.251ms, p99_ms=0.290ms
  op=matmul_bias_add | batch=1 | m=1 | n=1152 | k=896 | tA=0 | tB=0 | bias=1:
    backend=mlx: count=2280 | total=535.99ms | avg=0.235ms | min=0.196ms | max=1.834ms | p95_ms=0.244ms, p99_ms=0.278ms
  op=matmul | batch=1 | m=1 | n=896 | k=896 | tA=0 | tB=1 | strided_batch=false:
    backend=mlx: count=2280 | total=533.41ms | avg=0.234ms | min=0.188ms | max=1.495ms | p95_ms=0.243ms, p99_ms=0.277ms
  op=matmul | batch=1 | m=1 | n=151936 | k=896 | tA=0 | tB=1 | strided_batch=false:
    backend=mps: count=95 | total=238.14ms | avg=2.507ms | min=2.490ms | max=2.582ms | p95_ms=2.546ms, p99_ms=2.558ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=64 | k=65 | tA=0 | tB=0 | accumulate=1 | alpha=1.0000 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=5.38ms | avg=0.224ms | min=0.150ms | max=1.564ms | p95_ms=0.197ms, p99_ms=1.251ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=86 | k=64 | tA=0 | tB=1 | accumulate=1 | alpha=0.1250 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=5.19ms | avg=0.216ms | min=0.152ms | max=1.506ms | p95_ms=0.168ms, p99_ms=1.198ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=69 | k=64 | tA=0 | tB=1 | accumulate=1 | alpha=0.1250 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=5.04ms | avg=0.210ms | min=0.150ms | max=1.354ms | p95_ms=0.168ms, p99_ms=1.081ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=77 | k=64 | tA=0 | tB=1 | accumulate=1 | alpha=0.1250 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=4.82ms | avg=0.201ms | min=0.137ms | max=1.135ms | p95_ms=0.173ms, p99_ms=0.914ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=64 | k=59 | tA=0 | tB=0 | accumulate=1 | alpha=1.0000 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=4.81ms | avg=0.201ms | min=0.154ms | max=1.114ms | p95_ms=0.170ms, p99_ms=0.897ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=54 | k=64 | tA=0 | tB=1 | accumulate=1 | alpha=0.1250 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=4.23ms | avg=0.176ms | min=0.149ms | max=0.527ms | p95_ms=0.197ms, p99_ms=0.452ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=64 | k=81 | tA=0 | tB=0 | accumulate=1 | alpha=1.0000 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=4.21ms | avg=0.175ms | min=0.140ms | max=0.445ms | p95_ms=0.210ms, p99_ms=0.392ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=70 | k=64 | tA=0 | tB=1 | accumulate=1 | alpha=0.1250 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=4.21ms | avg=0.175ms | min=0.154ms | max=0.417ms | p95_ms=0.205ms, p99_ms=0.370ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=52 | k=64 | tA=0 | tB=1 | accumulate=1 | alpha=0.1250 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=4.17ms | avg=0.174ms | min=0.152ms | max=0.490ms | p95_ms=0.168ms, p99_ms=0.416ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=53 | k=64 | tA=0 | tB=1 | accumulate=1 | alpha=0.1250 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=4.08ms | avg=0.170ms | min=0.148ms | max=0.414ms | p95_ms=0.170ms, p99_ms=0.358ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=88 | k=64 | tA=0 | tB=1 | accumulate=1 | alpha=0.1250 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=4.03ms | avg=0.168ms | min=0.145ms | max=0.217ms | p95_ms=0.211ms, p99_ms=0.216ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=64 | k=91 | tA=0 | tB=0 | accumulate=1 | alpha=1.0000 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=4.00ms | avg=0.167ms | min=0.154ms | max=0.180ms | p95_ms=0.175ms, p99_ms=0.179ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=64 | k=68 | tA=0 | tB=0 | accumulate=1 | alpha=1.0000 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=4.00ms | avg=0.167ms | min=0.134ms | max=0.220ms | p95_ms=0.204ms, p99_ms=0.217ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=64 | k=95 | tA=0 | tB=0 | accumulate=1 | alpha=1.0000 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=4.00ms | avg=0.167ms | min=0.155ms | max=0.219ms | p95_ms=0.172ms, p99_ms=0.208ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=64 | k=90 | tA=0 | tB=0 | accumulate=1 | alpha=1.0000 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=3.99ms | avg=0.166ms | min=0.146ms | max=0.173ms | p95_ms=0.173ms, p99_ms=0.173ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=92 | k=64 | tA=0 | tB=1 | accumulate=1 | alpha=0.1250 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=3.99ms | avg=0.166ms | min=0.127ms | max=0.324ms | p95_ms=0.168ms, p99_ms=0.288ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=64 | k=94 | tA=0 | tB=0 | accumulate=1 | alpha=1.0000 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=3.98ms | avg=0.166ms | min=0.157ms | max=0.205ms | p95_ms=0.173ms, p99_ms=0.198ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=91 | k=64 | tA=0 | tB=1 | accumulate=1 | alpha=0.1250 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=3.98ms | avg=0.166ms | min=0.150ms | max=0.275ms | p95_ms=0.167ms, p99_ms=0.250ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=64 | k=73 | tA=0 | tB=0 | accumulate=1 | alpha=1.0000 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=3.97ms | avg=0.166ms | min=0.147ms | max=0.215ms | p95_ms=0.173ms, p99_ms=0.205ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=64 | k=71 | tA=0 | tB=0 | accumulate=1 | alpha=1.0000 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=3.97ms | avg=0.165ms | min=0.155ms | max=0.173ms | p95_ms=0.170ms, p99_ms=0.172ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=64 | k=67 | tA=0 | tB=0 | accumulate=1 | alpha=1.0000 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=3.96ms | avg=0.165ms | min=0.156ms | max=0.198ms | p95_ms=0.171ms, p99_ms=0.192ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=64 | k=53 | tA=0 | tB=0 | accumulate=1 | alpha=1.0000 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=3.96ms | avg=0.165ms | min=0.148ms | max=0.305ms | p95_ms=0.167ms, p99_ms=0.273ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=64 | k=70 | tA=0 | tB=0 | accumulate=1 | alpha=1.0000 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=3.95ms | avg=0.165ms | min=0.155ms | max=0.177ms | p95_ms=0.172ms, p99_ms=0.176ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=64 | k=84 | tA=0 | tB=0 | accumulate=1 | alpha=1.0000 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=3.95ms | avg=0.165ms | min=0.156ms | max=0.176ms | p95_ms=0.174ms, p99_ms=0.176ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=64 | k=76 | tA=0 | tB=0 | accumulate=1 | alpha=1.0000 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=3.95ms | avg=0.165ms | min=0.157ms | max=0.171ms | p95_ms=0.170ms, p99_ms=0.171ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=71 | k=64 | tA=0 | tB=1 | accumulate=1 | alpha=0.1250 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=3.94ms | avg=0.164ms | min=0.158ms | max=0.199ms | p95_ms=0.177ms, p99_ms=0.194ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=64 | k=89 | tA=0 | tB=0 | accumulate=1 | alpha=1.0000 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=3.94ms | avg=0.164ms | min=0.150ms | max=0.179ms | p95_ms=0.174ms, p99_ms=0.178ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=64 | k=92 | tA=0 | tB=0 | accumulate=1 | alpha=1.0000 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=3.94ms | avg=0.164ms | min=0.138ms | max=0.174ms | p95_ms=0.172ms, p99_ms=0.174ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=64 | k=88 | tA=0 | tB=0 | accumulate=1 | alpha=1.0000 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=3.94ms | avg=0.164ms | min=0.143ms | max=0.198ms | p95_ms=0.186ms, p99_ms=0.196ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=64 | k=79 | tA=0 | tB=0 | accumulate=1 | alpha=1.0000 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=3.94ms | avg=0.164ms | min=0.151ms | max=0.175ms | p95_ms=0.172ms, p99_ms=0.174ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=64 | k=87 | tA=0 | tB=0 | accumulate=1 | alpha=1.0000 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=3.93ms | avg=0.164ms | min=0.154ms | max=0.176ms | p95_ms=0.171ms, p99_ms=0.175ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=90 | k=64 | tA=0 | tB=1 | accumulate=1 | alpha=0.1250 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=3.93ms | avg=0.164ms | min=0.149ms | max=0.205ms | p95_ms=0.174ms, p99_ms=0.198ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=64 | k=66 | tA=0 | tB=0 | accumulate=1 | alpha=1.0000 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=3.93ms | avg=0.164ms | min=0.153ms | max=0.178ms | p95_ms=0.169ms, p99_ms=0.176ms
  op=matmul_alpha_beta_cache | batch=14 | m=1 | n=64 | k=69 | tA=0 | tB=0 | accumulate=1 | alpha=1.0000 | beta=0.0000 | cache=1 | strided_batch=true:
    backend=mlx: count=24 | total=3.93ms | avg=0.164ms | min=0.130ms | max=0.175ms | p95_ms=0.173ms, p99_ms=0.175ms
    