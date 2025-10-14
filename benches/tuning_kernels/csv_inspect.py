#!/usr/bin/env python3
"""
Quick CSV inspection: compares backends on matched Small-N shapes,
estimates dispatcher overhead, and summarizes direct GEMM (MPS vs MLX).

Usage:
  python benches/tuning_kernels/csv_inspect.py benches/benchmark_results.csv
"""

import sys
import re
import pandas as pd
import numpy as np


def parse_mkn(s: str):
    if not isinstance(s, str):
        return (np.nan, np.nan, np.nan)
    m = re.search(r"(\d+)x(\d+)x(\d+)", s)
    if not m:
        return (np.nan, np.nan, np.nan)
    return tuple(int(x) for x in m.groups())


def norm_variant(v: str):
    lv = v.lower() if isinstance(v, str) else ''
    lv = lv.replace(' ', '_')
    if lv in {'mps', 'mlx', 'gemv', 'auto', 'noop'}:
        return lv
    if 'via_dispatcher' in lv:
        return 'via_dispatcher'
    if 'direct_kernel' in lv:
        return 'direct_kernel'
    if 'smalln_direct_n' in lv:
        return lv  # keep specific small-n direct tag
    if 'gemm_direct_mps' in lv:
        return 'gemm_direct_mps'
    if 'gemm_direct_mlx' in lv:
        return 'gemm_direct_mlx'
    return lv


def load_csv(path):
    df = pd.read_csv(path)
    # Ensure numeric
    df['mean_time'] = pd.to_numeric(df['mean_time'], errors='coerce')
    return df.dropna(subset=['mean_time'])


def inspect_smalln(df):
    sdf = df[df['benchmark'].str.contains('smalln', case=False, na=False)].copy()
    if len(sdf) == 0:
        print("[Small-N] No rows found")
        return
    sdf['variant'] = sdf['variant'].apply(norm_variant)
    dims = sdf['parameters'].apply(parse_mkn)
    sdf[['m_dim', 'k_dim', 'n_dim']] = pd.DataFrame(dims.tolist(), index=sdf.index)
    sdf = sdf.dropna(subset=['m_dim', 'k_dim', 'n_dim'])
    sdf[['m_dim', 'k_dim', 'n_dim']] = sdf[['m_dim', 'k_dim', 'n_dim']].astype(int)

    print("\n[Small-N] Overall mean by backend (µs):")
    overall = sdf.groupby('variant', as_index=False)['mean_time'].mean()
    overall['mean_us'] = overall['mean_time'] / 1000.0
    for _, row in overall.sort_values('mean_us').iterrows():
        print(f"  {row['variant']:>20}: {row['mean_us']:.1f} µs")

    # Matched-shape comparisons on common shapes
    shapes = sdf.groupby(['m_dim', 'k_dim', 'n_dim']).size().sort_values(ascending=False)
    common_shapes = [(m, k, n) for (m, k, n), cnt in shapes.items() if cnt >= 3][:8]
    print("\n[Small-N] Matched-shape backend ranking (µs):")
    for m, k, n in common_shapes:
        slice_ = sdf[(sdf['m_dim'] == m) & (sdf['k_dim'] == k) & (sdf['n_dim'] == n)]
        # Aggregate to one value per backend
        agg = slice_.groupby('variant', as_index=False)['mean_time'].mean()
        agg['mean_us'] = agg['mean_time'] / 1000.0
        rank = agg.sort_values('mean_us')
        entries = ", ".join([f"{v}: {t:.1f}" for v, t in zip(rank['variant'], rank['mean_us'])])
        print(f"  {m}x{k}x{n} -> {entries}")


def inspect_dispatcher_overhead(df):
    vs = df[df['benchmark'].str.contains('smalln_vs_dispatcher', case=False, na=False)].copy()
    if len(vs) == 0:
        print("\n[Dispatcher Overhead] No comparison rows found")
        return
    vs['variant'] = vs['variant'].apply(norm_variant)
    dims = vs['parameters'].apply(parse_mkn)
    vs[['m_dim', 'k_dim', 'n_dim']] = pd.DataFrame(dims.tolist(), index=vs.index)

    print("\n[Dispatcher Overhead] Via vs Direct (and Noop if present) by shape (ratios, µs):")
    for params, grp in vs.groupby('parameters'):
        direct = grp[grp['variant'] == 'direct_kernel']['mean_time'].mean()
        via = grp[grp['variant'] == 'via_dispatcher']['mean_time'].mean()
        noop = grp[grp['variant'] == 'noop']['mean_time'].mean()
        if not np.isnan(direct) and direct > 0:
            m, k, n = parse_mkn(params)
            via_ratio = (via / direct) if not np.isnan(via) else np.nan
            noop_ratio = (noop / direct) if not np.isnan(noop) else np.nan
            via_str = f"via={via/1000.0:.1f} µs" if not np.isnan(via) else "via=NA"
            noop_str = f"noop={noop/1000.0:.1f} µs" if not np.isnan(noop) else "noop=NA"
            via_rstr = f"via/direct={via_ratio:.2f}" if not np.isnan(via_ratio) else "via/direct=NA"
            noop_rstr = f"noop/direct={noop_ratio:.2f}" if not np.isnan(noop_ratio) else "noop/direct=NA"
            print(f"  {m}x{k}x{n}: {via_str}, {noop_str}, direct={direct/1000.0:.1f} µs, {via_rstr}, {noop_rstr}")


def inspect_gemm_direct(df):
    gdf = df[df['benchmark'].str.contains('gemm_kernels_direct', case=False, na=False)].copy()
    if len(gdf) == 0:
        print("\n[Direct GEMM] No rows found")
        return
    gdf['variant'] = gdf['variant'].apply(norm_variant)
    dims = gdf['parameters'].apply(parse_mkn)
    gdf[['m_dim', 'k_dim', 'n_dim']] = pd.DataFrame(dims.tolist(), index=gdf.index)

    print("\n[Direct GEMM] MPS vs MLX on matched shapes (µs and ratio):")
    for (m, k, n), grp in gdf.groupby(['m_dim', 'k_dim', 'n_dim']):
        mps = grp[grp['variant'] == 'gemm_direct_mps']['mean_time'].mean()
        mlx = grp[grp['variant'] == 'gemm_direct_mlx']['mean_time'].mean()
        if not np.isnan(mps) and not np.isnan(mlx) and mlx > 0:
            ratio = mps / mlx
            print(f"  {m}x{k}x{n}: MPS={mps/1000.0:.1f} µs, MLX={mlx/1000.0:.1f} µs, MPS/MLX={ratio:.2f}")


def parse_softmax_params(s: str):
    if not isinstance(s, str):
        return (np.nan, np.nan)
    m = re.search(r"rows(\d+)_seqk(\d+)", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    # Direct kernel labels use seqq not rows
    m2 = re.search(r"seqq(\d+)_seqk(\d+)", s)
    if m2:
        return (int(m2.group(1)), int(m2.group(2)))
    return (np.nan, np.nan)


def inspect_softmax(df):
    sdf = df[df['benchmark'].str.contains('softmax', case=False, na=False)].copy()
    if len(sdf) == 0:
        print("\n[Softmax] No rows found")
        return
    # Normalize variants of interest
    def norm_softmax_variant(v: str):
        if not isinstance(v, str):
            return v
        lv = v.lower()
        if 'softmax_mps' in lv:
            return 'mps'
        if 'softmax_kernel' in lv:
            return 'kernel'
        if 'softmax_auto' in lv:
            return 'auto'
        if 'softmax_noop' in lv or (lv.startswith('softmax') and 'noop' in lv):
            return 'noop'
        if 'softmaxdispatch_vec' in lv:
            return 'vec'
        if 'softmaxdispatch_block' in lv:
            return 'block'
        if 'softmax_causal' in lv:
            return 'causal'
        if 'softmax_normal' in lv:
            return 'normal'
        return v
    sdf['variant'] = sdf['variant'].apply(norm_softmax_variant)
    dims = sdf['parameters'].apply(parse_softmax_params)
    sdf[['rows_total', 'seq_k']] = pd.DataFrame(dims.tolist(), index=sdf.index)
    sdf = sdf.dropna(subset=['rows_total', 'seq_k'])
    sdf[['rows_total', 'seq_k']] = sdf[['rows_total', 'seq_k']].astype(int)

    # Focus on dispatcher seq + mps + kernel comparisons at fixed rows_total
    # Choose common rows_total = 128 for clarity
    rows_val = 128
    slice_ = sdf[sdf['rows_total'] == rows_val].copy()
    if len(slice_) == 0:
        # fallback to most common rows
        common_rows = sdf['rows_total'].mode()
        if len(common_rows):
            rows_val = int(common_rows.iloc[0])
            slice_ = sdf[sdf['rows_total'] == rows_val].copy()

    print(f"\n[Softmax] Matched rows={rows_val} rankings by seq_k (µs):")
    for k, grp in slice_.groupby('seq_k'):
        agg = grp.groupby('variant', as_index=False)['mean_time'].mean()
        agg['mean_us'] = agg['mean_time'] / 1000.0
        rank = agg.sort_values('mean_us')
        entries = ", ".join([f"{v}: {t:.1f}" for v, t in zip(rank['variant'], rank['mean_us'])])
        print(f"  seq_k={k}: {entries}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python benches/tuning_kernels/csv_inspect.py <csv_path>")
        sys.exit(2)
    path = sys.argv[1]
    df = load_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    inspect_softmax(df)
    inspect_smalln(df)
    inspect_dispatcher_overhead(df)
    inspect_gemm_direct(df)


if __name__ == '__main__':
    main()