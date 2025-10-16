#!/usr/bin/env python3
"""
Collect Criterion benchmark results into a single CSV for analysis.

This script walks the `target/criterion` directory, reads `new/estimates.json`
files, and emits a CSV with columns:
- benchmark: top-level bench group name (e.g., softmax_dispatcher_seq_f16)
- parameters: sub-benchmark identifier (e.g., seq1024_head128 or 128x1024x1_a1_b0)
- mean_time: Criterion mean point estimate (nanoseconds)
- throughput: 0 (placeholder; Criterion throughput not parsed)

Usage:
  python benches/collect_criterion_csv.py --criterion-dir target/criterion --out benches/benchmark_results.csv
"""

import argparse
import json
import csv
import os
import re
from pathlib import Path


def collect_rows(criterion_dir: Path):
    rows = []

    def extract_meta(estimates_path: Path):
        # Expect .../<group>/<maybe-variant>/<params>/new/estimates.json
        parts = estimates_path.parts
        try:
            new_idx = parts.index("new")
        except ValueError:
            return None
        params_name = parts[new_idx - 1] if new_idx - 1 >= 0 else ""
        variant_or_group = parts[new_idx - 2] if new_idx - 2 >= 0 else ""
        group_maybe = parts[new_idx - 3] if new_idx - 3 >= 0 else ""

        # Determine benchmark group name; skip known non-bench roots
        benchmark = group_maybe if group_maybe not in {"data", "main"} else variant_or_group
        # Infer variant, handling both standard and data/main layouts
        if group_maybe not in {"data", "main"}:
            variant = variant_or_group
        else:
            variant = ""
            # Try scanning for variant tokens
            for p in parts:
                if re.search(r"SoftmaxDispatch_(vec|block)", p):
                    variant = p
                    break
                if re.search(r"Softmax_(auto|kernel|mps)", p):
                    variant = p
                    break
                if p in {"mlx", "mps", "gemv", "gemm_tiled", "auto"}:
                    variant = p
                    break

        # Normalize params for small-N matmul
        # Preserve full parameters for richer analysis (retain case labels)
        params_norm = params_name
        return benchmark, variant, params_norm

    for dirpath, dirnames, filenames in os.walk(criterion_dir):
        if os.path.basename(dirpath) != "new":
            continue
        if "estimates.json" not in filenames:
            continue
        est_path = Path(dirpath) / "estimates.json"
        meta = extract_meta(est_path)
        if not meta:
            continue
        benchmark, variant, params = meta
        try:
            with est_path.open("r") as f:
                data = json.load(f)
            mean_ns = float(data.get("mean", {}).get("point_estimate"))
        except Exception:
            mean_ns = 0.0
        # Compute throughput based on benchmark type and parameter label semantics
        def compute_throughput(bmk: str, params_label: str, mean_ns_val: float) -> float:
            # Handle negative or zero mean values properly
            if not mean_ns_val or mean_ns_val <= 0:
                print(f"WARNING: Invalid mean_ns_val: {mean_ns_val} for benchmark {bmk}, params {params_label}")
                return 0.0
            
            # Decide op factor: softmax ~3 ops per element, matmul ~2 ops (MAC)
            op_factor = 3.0 if "softmax" in bmk.lower() else 2.0

            # Try parsing known label formats
            elems = 0
            # Softmax 2D new: rows{R}_seqk{K}
            m = re.match(r"^rows(\d+)_seqk(\d+)$", params_label)
            if m:
                R, K = int(m.group(1)), int(m.group(2))
                elems = R * K
            # Softmax 3D new: batch{B}_seqq{Q}_seqk{K}
            m = re.match(r"^batch(\d+)_seqq(\d+)_seqk(\d+)$", params_label)
            if m and elems == 0:
                B, Q, K = int(m.group(1)), int(m.group(2)), int(m.group(3))
                elems = B * Q * K
            # Softmax older 2D: seq{S}_head{H}
            m = re.match(r"^seq(\d+)_head(\d+)$", params_label)
            if m and elems == 0:
                S, H = int(m.group(1)), int(m.group(2))
                elems = S * H
            # Softmax older 3D: batch{B}_seq{S}_head{H}
            m = re.match(r"^batch(\d+)_seq(\d+)_head(\d+)$", params_label)
            if m and elems == 0:
                B, S, H = int(m.group(1)), int(m.group(2)), int(m.group(3))
                elems = B * S * H
            # Softmax direct kernels: seqq{Q}_seqk{K}
            m = re.match(r"^seqq(\d+)_seqk(\d+)$", params_label)
            if m and elems == 0:
                Q, K = int(m.group(1)), int(m.group(2))
                elems = Q * K

            # Matmul: MxKxN_a*
            m = re.match(r"^(\d+)x(\d+)x(\d+)_a.*$", params_label)
            if m and elems == 0:
                M, K, N = int(m.group(1)), int(m.group(2)), int(m.group(3))
                elems = M * K * N
            # Matmul tuning with threshold suffix
            m = re.match(r"^(\d+)x(\d+)x(\d+)thresh(\d+)$", params_label)
            if m and elems == 0:
                M, K, N = int(m.group(1)), int(m.group(2)), int(m.group(3))
                elems = M * K * N
            # Matmul simple MxKxN (no suffix)
            m = re.match(r"^(\d+)x(\d+)x(\d+)$", params_label)
            if m and elems == 0:
                M, K, N = int(m.group(1)), int(m.group(2)), int(m.group(3))
                elems = M * K * N
            # Matmul batched: batchB_MxKxN
            m = re.match(r"^batch(\d+)_(\d+)x(\d+)x(\d+)$", params_label)
            if m and elems == 0:
                B, M, K, N = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
                elems = B * M * K * N

            if elems == 0:
                print(f"WARNING: Could not parse parameter format: {params_label} for benchmark {bmk}")
                return 0.0
                
            # Calculate throughput: operations / time in seconds
            # mean_ns_val is in nanoseconds, convert to seconds by dividing by 1e9
            time_seconds = mean_ns_val / 1e9
            if time_seconds <= 0:
                print(f"ERROR: Invalid time calculation: mean_ns_val={mean_ns_val}, time_seconds={time_seconds}")
                return 0.0
                
            ops_per_second = float(elems * op_factor) / time_seconds
            return ops_per_second

        rows.append({
            "benchmark": benchmark,
            "variant": variant,
            "parameters": params,
            "mean_time": mean_ns,
            "throughput": compute_throughput(benchmark, params, mean_ns),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Collect Criterion results into CSV")
    parser.add_argument("--criterion-dir", default="target/criterion", help="Path to Criterion output directory")
    parser.add_argument("--out", default="benches/benchmark_results.csv", help="Output CSV path")

    args = parser.parse_args()
    criterion_dir = Path(args.criterion_dir)
    out_path = Path(args.out)

    if not criterion_dir.exists():
        print(f"Error: criterion directory not found: {criterion_dir}")
        return 1

    rows = collect_rows(criterion_dir)
    # Also collect from Criterion's data/main path where some benches are stored
    data_main = criterion_dir / "data" / "main"
    if data_main.exists():
        rows += collect_rows(data_main)
    if not rows:
        print("No benchmark results found in Criterion directory.")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["benchmark", "variant", "parameters", "mean_time", "throughput"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
