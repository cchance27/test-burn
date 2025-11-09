These are the sizes and shapes from a qwen25 inference run with our metallic inference rust framework.

The times are from running with METALLIC_PROFILING_ENABLED and are likely CPU timed not jsut the GPU timed.

Matmul backend summary:
  backend=mlx: count=9120 | total=1794.58ms | avg=0.197ms | min=0.105ms | max=2.613ms | p95_ms=0.241ms, p99_ms=0.282ms
  backend=mps: count=4655 | total=1524.18ms | avg=0.327ms | min=0.193ms | max=7.734ms | p95_ms=0.346ms, p99_ms=2.505ms

Matmul op/backend summary:
  matmul @ mps: count=4655 | total=1524.18ms | avg=0.327ms | min=0.193ms | max=7.734ms | p95_ms=0.346ms, p99_ms=2.505ms
  matmul_alpha_beta @ mlx: count=4560 | total=725.13ms | avg=0.159ms | min=0.105ms | max=2.613ms | p95_ms=0.169ms, p99_ms=0.214ms
  matmul @ mlx: count=2280 | total=534.83ms | avg=0.235ms | min=0.160ms | max=1.674ms | p95_ms=0.244ms, p99_ms=0.289ms
  matmul_bias_add @ mlx: count=2280 | total=534.62ms | avg=0.234ms | min=0.161ms | max=1.613ms | p95_ms=0.244ms, p99_ms=0.292ms

Matmul shape backend summary:
  op=matmul | batch=1 | m=1 | n=9728 | k=896 | tA=0 | tB=1:
    backend=mps: count=2280 | total=740.54ms | avg=0.325ms | min=0.261ms | max=7.734ms | p95_ms=0.347ms, p99_ms=0.416ms
  op=matmul | batch=1 | m=1 | n=896 | k=4864 | tA=0 | tB=1:
    backend=mps: count=2280 | total=545.15ms | avg=0.239ms | min=0.193ms | max=1.355ms | p95_ms=0.252ms, p99_ms=0.298ms
  op=matmul | batch=1 | m=1 | n=896 | k=896 | tA=0 | tB=1:
    backend=mlx: count=2280 | total=534.83ms | avg=0.235ms | min=0.160ms | max=1.674ms | p95_ms=0.244ms, p99_ms=0.289ms
  op=matmul_bias_add | batch=1 | m=1 | n=1152 | k=896 | tA=0 | tB=0:
    backend=mlx: count=2280 | total=534.62ms | avg=0.234ms | min=0.161ms | max=1.613ms | p95_ms=0.244ms, p99_ms=0.292ms
  op=matmul | batch=1 | m=1 | n=151936 | k=896 | tA=0 | tB=1:
    backend=mps: count=95 | total=238.50ms | avg=2.510ms | min=2.448ms | max=2.715ms | p95_ms=2.533ms, p99_ms=2.661ms
  op=matmul_alpha_beta | batch=14 | m=1 | n=64 | k=95 | tA=0 | tB=0:
    backend=mlx: count=24 | total=6.00ms | avg=0.250ms | min=0.138ms | max=2.205ms | p95_ms=0.183ms, p99_ms=1.740ms
  op=matmul_alpha_beta | batch=14 | m=1 | n=64 | k=16 | tA=0 | tB=0:
    backend=mlx: count=24 | total=5.96ms | avg=0.249ms | min=0.121ms | max=2.613ms | p95_ms=0.156ms, p99_ms=2.048ms
  op=matmul_alpha_beta | batch=14 | m=1 | n=51 | k=64 | tA=0 | tB=1:
    backend=mlx: count=24 | total=5.54ms | avg=0.231ms | min=0.149ms | max=1.810ms | p95_ms=0.205ms, p99_ms=1.442ms
  op=matmul_alpha_beta | batch=14 | m=1 | n=64 | k=88 | tA=0 | tB=0: