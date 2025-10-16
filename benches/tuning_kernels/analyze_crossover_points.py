#!/usr/bin/env python3
"""
Benchmark Analysis Script for 5.2 Kernel Optimization

This script analyzes benchmark results to determine optimal crossover points
between different kernel variants (vec-softmax vs block-softmax, Small-N GEMV variants).

Usage:
    python analyze_crossover_points.py benchmark_results.csv

The script will output:
- Optimal crossover thresholds for different hardware configurations
- Performance recommendations for dispatcher tuning
- Analysis of kernel selection effectiveness
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys

def load_benchmark_data(csv_path):
    """Load and preprocess benchmark data."""
    if not Path(csv_path).exists():
        print(f"Error: Benchmark results file not found: {csv_path}")
        print("Please run the benchmarks first to generate performance data.")
        return None

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} benchmark results from {csv_path}")
    return df

def analyze_softmax_crossover(df):
    """Analyze optimal crossover point between vec-softmax and block-softmax."""
    print("\n" + "="*60)
    print("SOFTMAX CROSSOVER ANALYSIS")
    print("="*60)

    # Filter for softmax dispatcher benchmarks
    softmax_df = df[df['benchmark'].str.contains('softmax', case=False, na=False)].copy()
    # Prefer analyzing f16 to keep lines comparable; adjust as needed
    softmax_df = softmax_df[softmax_df['benchmark'].str.contains('f16', case=False, na=False)]

    if len(softmax_df) == 0:
        print("No softmax benchmark data found.")
        return None

    # Extract keys sequence length (seq_k) from parameters; support new and legacy labels
    seqk_series = softmax_df['parameters'].str.extract(r'seqk(\d+)')[0]
    seqlen_legacy = softmax_df['parameters'].str.extract(r'seq(\d+)')[0]
    # Prefer seq_k when available, otherwise fall back to legacy seq
    softmax_df['seq_len'] = pd.to_numeric(seqk_series.fillna(seqlen_legacy), errors='coerce').astype('Int64')
    softmax_df = softmax_df.dropna(subset=['seq_len'])

    # Group by sequence length and variant (mps, kernel, vec, block)
    crossover_data = []
    for seq_len in sorted(softmax_df['seq_len'].unique()):
        seq_data = softmax_df[softmax_df['seq_len'] == seq_len]
        variant_specs = [
            ('vec', r'^SoftmaxDispatch_vec$'),
            ('block', r'^SoftmaxDispatch_block$'),
            ('mps', r'^Softmax_mps$'),
            ('kernel', r'^Softmax_kernel$'),
        ]
        for label, pattern in variant_specs:
            if 'variant' in seq_data.columns:
                variant_data = seq_data[seq_data['variant'].str.contains(pattern, case=True, na=False)]
            else:
                variant_data = seq_data[seq_data['benchmark'].str.contains(pattern, case=False, na=False)]
            if len(variant_data) > 0:
                mean_time = variant_data['mean_time'].mean()
                crossover_data.append({
                    'seq_len': seq_len,
                    'variant': label,
                    'mean_time': mean_time,
                    'throughput': variant_data['throughput'].iloc[0] if 'throughput' in variant_data.columns else 0
                })

    if len(crossover_data) == 0:
        print("No crossover data found.")
        return None

    crossover_df = pd.DataFrame(crossover_data)

    # Build causal vs normal comparison dataset from dedicated causal softmax benchmarks (f16 only)
    modes_rows = []
    causal_df = df[df['benchmark'].str.contains('softmax_dispatcher_causal', case=False, na=False)].copy()
    causal_df = causal_df[causal_df['benchmark'].str.contains('f16', case=False, na=False)]
    if len(causal_df) > 0:
        # Extract keys length; support new labels rows{R}_seqk{K} and legacy seq{S}
        seqk_series = causal_df['parameters'].str.extract(r'seqk(\d+)')[0]
        seq_series = causal_df['parameters'].str.extract(r'seq(\d+)')[0]
        causal_df['seq_len'] = pd.to_numeric(seqk_series.fillna(seq_series), errors='coerce').astype('Int64')
        causal_df = causal_df.dropna(subset=['seq_len'])
        for seq_len in sorted(causal_df['seq_len'].unique()):
            seq_data = causal_df[causal_df['seq_len'] == seq_len]
            for mode_label, pattern in [('causal', r'^Softmax_Causal$'), ('normal', r'^Softmax_Normal$')]:
                if 'variant' in seq_data.columns:
                    vdf = seq_data[seq_data['variant'].str.contains(pattern, case=True, na=False)]
                else:
                    vdf = seq_data[seq_data['benchmark'].str.contains(pattern, case=False, na=False)]
                if len(vdf) > 0:
                    modes_rows.append({
                        'seq_len': seq_len,
                        'mode': mode_label,
                        'mean_time': vdf['mean_time'].mean(),
                        'throughput': vdf['throughput'].iloc[0] if 'throughput' in vdf.columns else 0
                    })
    modes_df = pd.DataFrame(modes_rows) if len(modes_rows) > 0 else None

    # Find crossover points where block becomes faster than vec
    crossover_points = []
    for seq_len in sorted(crossover_df['seq_len'].unique()):
        seq_data = crossover_df[crossover_df['seq_len'] == seq_len]

        vec_time = seq_data[seq_data['variant'] == 'vec']['mean_time']
        block_time = seq_data[seq_data['variant'] == 'block']['mean_time']

        if len(vec_time) > 0 and len(block_time) > 0:
            vec_time = vec_time.iloc[0]
            block_time = block_time.iloc[0]

            if block_time < vec_time:
                crossover_points.append(int(seq_len))

    # Find optimal crossover point (where the difference is most significant)
    optimal_crossover = None
    max_improvement = 0

    for seq_len in crossover_points:
        seq_data = crossover_df[crossover_df['seq_len'] == seq_len]
        vec_time = seq_data[seq_data['variant'] == 'vec']['mean_time'].iloc[0]
        block_time = seq_data[seq_data['variant'] == 'block']['mean_time'].iloc[0]

        improvement = (vec_time - block_time) / vec_time
        if improvement > max_improvement:
            max_improvement = improvement
            optimal_crossover = int(seq_len)

    best_variants = (
        crossover_df.sort_values('mean_time')
        .drop_duplicates('seq_len', keep='first')
        .sort_values('seq_len')
    )

    if len(best_variants) > 0:
        print("\n[Softmax] Best variant per sequence length:")
        for _, row in best_variants.iterrows():
            print(f"  seq_len={int(row['seq_len'])}: {row['variant']} ({row['mean_time']/1000.0:.1f} µs)")

    transitions = []
    if len(best_variants) > 0:
        prev_variant = None
        for _, row in best_variants.iterrows():
            variant = row['variant']
            seq_len = int(row['seq_len'])
            if variant != prev_variant:
                transitions.append({'start': seq_len, 'variant': variant})
                prev_variant = variant

        if transitions:
            print("\n[Softmax] Suggested dispatcher regions:")
            for idx, entry in enumerate(transitions):
                start = entry['start']
                variant = entry['variant']
                if idx + 1 < len(transitions):
                    end = transitions[idx + 1]['start'] - 1
                    print(f"  {variant}: seq_len ∈ [{start}, {end}]")
                else:
                    print(f"  {variant}: seq_len ≥ {start}")

    print(f"\nFound {len(crossover_points)} potential crossover points: {crossover_points}")
    print(f"Optimal crossover point: {optimal_crossover} (improvement: {max_improvement:.2%})")

    return {
        'optimal_crossover': optimal_crossover,
        'crossover_points': crossover_points,
        'data': crossover_df,
        'modes': modes_df,
        'best_variants': best_variants.to_dict('records'),
        'transitions': transitions,
    }

def analyze_smalln_gemv_crossover(df):
    """Analyze optimal N thresholds for Small-N GEMV kernels."""
    print("\n" + "="*60)
    print("SMALL-N GEMV CROSSOVER ANALYSIS")
    print("="*60)

    # Filter for Small-N GEMV benchmarks
    smalln_df = df[df['benchmark'].str.contains('smalln', case=False, na=False)].copy()
    # Prefer analyzing f16 to keep lines comparable
    smalln_df = smalln_df[smalln_df['benchmark'].str.contains('f16', case=False, na=False)]

    if len(smalln_df) == 0:
        print("No Small-N GEMV benchmark data found.")
        return None

    # Extract dimensions from parameters; support legacy 'nN' and extended 'MxKxN_case'
    n_primary = pd.to_numeric(smalln_df['parameters'].str.extract(r'n(\d+)')[0], errors='coerce')
    dims = smalln_df['parameters'].str.extract(r'(?P<m_dim>\d+)x(?P<k_dim>\d+)x(?P<n_dim_mkn>\d+)')
    # Use MxKxN parse if available
    smalln_df['m_dim'] = pd.to_numeric(dims['m_dim'], errors='coerce')
    smalln_df['k_dim'] = pd.to_numeric(dims['k_dim'], errors='coerce')
    n_fallback = pd.to_numeric(dims['n_dim_mkn'], errors='coerce')
    smalln_df['n_dim'] = n_primary
    mask_missing = smalln_df['n_dim'].isna()
    smalln_df.loc[mask_missing, 'n_dim'] = n_fallback[mask_missing]
    smalln_df['n_dim'] = pd.to_numeric(smalln_df['n_dim'], errors='coerce')
    smalln_df = smalln_df.dropna(subset=['n_dim'])
    smalln_df['n_dim'] = smalln_df['n_dim'].astype(int)
    # Parse alpha/beta case labels if present, default to a1_b0
    smalln_df['case'] = smalln_df['parameters'].str.extract(r'_(a\d+_b\d+)')
    smalln_df['case'] = smalln_df['case'].fillna('a1_b0')

    # Compute per-backend performance vs N
    backend_variants = sorted(smalln_df['variant'].unique()) if 'variant' in smalln_df.columns else []
    # Normalize variant labels for plotting clarity
    def norm_variant(v):
        if isinstance(v, str):
            lv = v.strip().lower()
            if lv == 'gemv':
                return 'gemv'
            if v.startswith('SmallN_Direct_N'):
                # Map to explicit direct small-n variant label
                return f"gemv_direct_{v.split('_')[-1].lower()}"  # e.g., gemv_direct_n8
            if v == 'SmallN_Direct':
                return 'gemv_direct'
            if lv == 'direct_kernel':
                return 'direct_kernel'
            if lv == 'via_dispatcher':
                return 'via_dispatcher'
            if lv == 'gemm_tiled':
                return 'gemm_tiled'
            if lv == 'mlx':
                return 'mlx'
            if lv == 'mps':
                return 'mps'
            if lv == 'noop':
                return 'noop'
            if lv == 'auto':
                return 'auto'
            if lv == 'gemv_direct':
                return 'gemv_direct'
            if lv.startswith('gemm_direct_tiled'):
                return 'gemm_direct_tiled'
            return lv
        return v
    perf_rows = []
    for case in sorted(smalln_df['case'].unique()):
        case_df = smalln_df[smalln_df['case'] == case]
        for n in sorted(case_df['n_dim'].unique()):
            n_slice = case_df[case_df['n_dim'] == n]
            if backend_variants:
                for backend in backend_variants:
                    b_slice = n_slice[n_slice['variant'] == backend]
                    if len(b_slice) == 0:
                        continue
                    perf_rows.append({
                        'n_dim': n,
                        'variant': norm_variant(backend),
                        'case': case,
                        'm_dim': pd.to_numeric(b_slice['m_dim'], errors='coerce').mean(),
                        'k_dim': pd.to_numeric(b_slice['k_dim'], errors='coerce').mean(),
                        'mean_time': b_slice['mean_time'].mean(),
                        'throughput': b_slice['throughput'].iloc[0] if 'throughput' in b_slice.columns else 0
                    })
            else:
                perf_rows.append({
                    'n_dim': n,
                    'variant': 'auto',
                    'case': case,
                    'm_dim': pd.to_numeric(n_slice['m_dim'], errors='coerce').mean(),
                    'k_dim': pd.to_numeric(n_slice['k_dim'], errors='coerce').mean(),
                    'mean_time': n_slice['mean_time'].mean(),
                    'throughput': n_slice['throughput'].iloc[0] if 'throughput' in n_slice.columns else 0
                })

    perf_df = pd.DataFrame(perf_rows)
    if len(perf_df) == 0:
        print("No Small-N performance rows found after parsing parameters.")
        return {
            'data': perf_df,
            'optimal_thresholds': [],
            'backends': backend_variants,
            'best_variants': [],
            'transitions': [],
        }

    perf_df = perf_df[perf_df['variant'] != 'noop']
    backend_variants = sorted(perf_df['variant'].unique())
    # Aggregate per N across variants
    avg_perf = (
        perf_df.groupby(['n_dim', 'variant'], as_index=False)['mean_time']
        .mean()
        .sort_values(['n_dim', 'mean_time'])
    )
    best_per_n = (
        avg_perf.sort_values('mean_time')
        .drop_duplicates('n_dim', keep='first')
        .sort_values('n_dim')
    )

    if len(best_per_n) > 0:
        print("\n[Small-N] Best backend per N:")
        for _, row in best_per_n.iterrows():
            print(f"  N={int(row['n_dim'])}: {row['variant']} ({row['mean_time']/1000.0:.1f} µs)")

    smalln_transitions = []
    if len(best_per_n) > 0:
        prev_variant = None
        for _, row in best_per_n.iterrows():
            variant = row['variant']
            n_dim = int(row['n_dim'])
            if variant != prev_variant:
                smalln_transitions.append({'start': n_dim, 'variant': variant})
                prev_variant = variant

        if smalln_transitions:
            print("\n[Small-N] Suggested dispatcher regions:")
            for idx, entry in enumerate(smalln_transitions):
                start = entry['start']
                variant = entry['variant']
                if idx + 1 < len(smalln_transitions):
                    end = smalln_transitions[idx + 1]['start'] - 1
                    print(f"  {variant}: N ∈ [{start}, {end}]")
                else:
                    print(f"  {variant}: N ≥ {start}")

    threshold_breaks = [entry['start'] for entry in smalln_transitions[1:]]

    print(f"\nSmall-N entries: {len(smalln_df)} rows; N values: {sorted(smalln_df['n_dim'].unique().tolist())}")
    if backend_variants:
        print(f"Backends found: {backend_variants}")
    if threshold_breaks:
        print(f"Proposed Small-N thresholds: {threshold_breaks}")

    return {
        'data': perf_df,
        'optimal_thresholds': threshold_breaks,
        'backends': backend_variants,
        'best_variants': best_per_n.to_dict('records'),
        'transitions': smalln_transitions,
    }

def generate_recommendations(softmax_analysis, smalln_analysis):
    """Generate tuning recommendations based on analysis."""
    print("\n" + "="*60)
    print("TUNING RECOMMENDATIONS")
    print("="*60)

    recommendations = []

    if softmax_analysis:
        transitions = softmax_analysis.get('transitions', []) or []
        if len(transitions) > 1:
            threshold = int(transitions[1]['start'])
            recommendations.append(f"Set METALLIC_SOFTMAX_VEC_BLOCK_THRESHOLD={threshold}")
            recommendations.append(f"Use {transitions[0]['variant']} below {threshold}, switch to {transitions[1]['variant']} at ≥{threshold}")
        else:
            optimal = softmax_analysis.get('optimal_crossover')
            if optimal:
                recommendations.append(f"Set METALLIC_SOFTMAX_VEC_BLOCK_THRESHOLD={int(optimal)}")
            else:
                recommendations.append("Keep current softmax threshold (1024) - no clear optimal found")
        if transitions:
            regions = []
            for idx, entry in enumerate(transitions):
                start = entry['start']
                variant = entry['variant']
                if idx + 1 < len(transitions):
                    end = transitions[idx + 1]['start'] - 1
                    regions.append(f"{variant}: seq_len ∈ [{start}, {end}]")
                else:
                    regions.append(f"{variant}: seq_len ≥ {start}")
            recommendations.append("Softmax backend regions → " + "; ".join(regions))

    if smalln_analysis:
        thresholds = smalln_analysis.get('optimal_thresholds', []) or []
        transitions = smalln_analysis.get('transitions', []) or []
        if thresholds:
            recommendations.append(f"Consider Small-N thresholds at: {thresholds}")
        else:
            recommendations.append("Keep current Small-N threshold (8) - good performance across N values")
        if transitions:
            segments = []
            for idx, entry in enumerate(transitions):
                start = entry['start']
                variant = entry['variant']
                if idx + 1 < len(transitions):
                    end = transitions[idx + 1]['start'] - 1
                    segments.append(f"{variant}: N ∈ [{start}, {end}]")
                else:
                    segments.append(f"{variant}: N ≥ {start}")
            recommendations.append("Small-N backend regions → " + "; ".join(segments))

    recommendations.append("Run benchmarks on target hardware for precise tuning")
    recommendations.append("Consider different thresholds for different GPU families (M1/M2/M3)")

    for rec in recommendations:
        print(f"• {rec}")

    return recommendations

def create_visualization(softmax_analysis, smalln_analysis, gemm_analysis=None, gemm_disp_analysis=None, output_dir="benchmark_analysis"):
    """Create visualizations of the benchmark analysis."""
    Path(output_dir).mkdir(exist_ok=True)

    # Softmax crossover visualization
    if softmax_analysis and 'data' in softmax_analysis:
        plt.figure(figsize=(10, 6))

        df = softmax_analysis['data']
        for variant, label in [('vec', 'vec-softmax'), ('block', 'block-softmax'), ('mps', 'mps'), ('kernel', 'kernel')]:
            variant_data = df[df['variant'] == variant]
            if len(variant_data) > 0:
                plt.plot(variant_data['seq_len'], variant_data['mean_time'] / 1000.0,
                        marker='o', label=label, linewidth=2)

        plt.xlabel('Sequence Length')
        plt.ylabel('Mean Time (µs)')
        plt.title('Softmax Kernel Performance vs Sequence Length')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ticklabel_format(style='plain', axis='y')
        plt.savefig(f'{output_dir}/softmax_crossover.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Small-N GEMV visualization (aligned domain)
    if smalln_analysis and 'data' in smalln_analysis:
        df = smalln_analysis['data']
        if len(df) > 0 and 'n_dim' in df.columns:
            fixed_n = [1, 2, 4, 8, 16]
            plt.figure(figsize=(10, 6))
            if 'variant' in df.columns:
                for backend in sorted(df['variant'].unique()):
                    bdf = df[df['variant'] == backend]
                    # Aggregate across cases/shapes to ensure unique N labels
                    bdf_agg = bdf.groupby('n_dim', as_index=False)['mean_time'].mean()
                    series = (bdf_agg.set_index('n_dim')['mean_time'] / 1000.0).reindex(fixed_n)
                    label = backend if backend != 'gemv_direct' else 'gemv-direct'
                    plt.plot(fixed_n, series.values, marker='o', linewidth=2, label=label)
            else:
                df_agg = df.groupby('n_dim', as_index=False)['mean_time'].mean()
                series = (df_agg.set_index('n_dim')['mean_time'] / 1000.0).reindex(fixed_n)
                plt.plot(fixed_n, series.values, marker='o', linewidth=2, color='blue', label='auto')

            plt.xticks(fixed_n)
            plt.xlabel('N Dimension')
            plt.ylabel('Mean Time (µs)')
            plt.title('Small-N GEMV Performance vs N Dimension (aligned domain)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ticklabel_format(style='plain', axis='y')
            plt.savefig(f'{output_dir}/smalln_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print("Skipping Small-N plot: no parsed data available.")

    # Small-N matched-shape overlays: compare backends on identical (M,K)
    if smalln_analysis and 'data' in smalln_analysis:
        df = smalln_analysis['data']
        if len(df) > 0 and set(['variant', 'n_dim', 'm_dim', 'k_dim']).issubset(df.columns):
            fixed_n = [1, 2, 4, 8, 16]
            # Clean up possible non-integer m/k
            df['m_dim'] = pd.to_numeric(df['m_dim'], errors='coerce')
            df['k_dim'] = pd.to_numeric(df['k_dim'], errors='coerce')
            df_shapes = df.dropna(subset=['m_dim', 'k_dim']).copy()
            # Aggregate mean_time per (variant, m_dim, k_dim, n_dim)
            agg = df_shapes.groupby(['variant', 'm_dim', 'k_dim', 'n_dim'], as_index=False)['mean_time'].mean()
            # For each shape, plot overlay of all backends
            for (m_dim, k_dim), sdf in agg.groupby(['m_dim', 'k_dim']):
                plt.figure(figsize=(10, 6))
                backends = sorted(sdf['variant'].unique())
                plotted = False
                for backend in backends:
                    line = sdf[sdf['variant'] == backend].sort_values('n_dim')
                    series = (line.set_index('n_dim')['mean_time'] / 1000.0).reindex(fixed_n)
                    label = backend if backend != 'gemv_direct' else 'gemv-direct'
                    plt.plot(fixed_n, series.values, marker='o', linewidth=2, label=label)
                    plotted = True
                if not plotted:
                    plt.close()
                    continue
                plt.xticks(fixed_n)
                plt.xlabel('N Dimension')
                plt.ylabel('Mean Time (µs)')
                plt.title(f'Small-N GEMV — Matched Shape Overlay {int(m_dim)}x{int(k_dim)}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.ticklabel_format(style='plain', axis='y')
                plt.savefig(f"{output_dir}/smalln_matched_overlay_{int(m_dim)}x{int(k_dim)}.png", dpi=300, bbox_inches='tight')
                plt.close()

    # Additional visualization: Softmax causal vs normal comparison if available
    if softmax_analysis and softmax_analysis.get('modes') is not None:
        modes_df = softmax_analysis['modes']
        if modes_df is not None and len(modes_df) > 0:
            plt.figure(figsize=(10, 6))
            for mode in sorted(modes_df['mode'].unique()):
                mdf = modes_df[modes_df['mode'] == mode].sort_values('seq_len')
                plt.plot(mdf['seq_len'], mdf['mean_time'], marker='o', linewidth=2, label=mode)
            plt.xlabel('Sequence Length')
            plt.ylabel('Mean Time (ns)')
            plt.title('Softmax Performance: Causal vs Normal (f16)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{output_dir}/softmax_modes.png', dpi=300, bbox_inches='tight')
            plt.close()

    # Additional visualization: Small-N per-case faceted plots (aligned domain)
    if smalln_analysis and 'data' in smalln_analysis:
        df = smalln_analysis['data']
        if len(df) > 0 and 'case' in df.columns and 'n_dim' in df.columns:
            for case in sorted(df['case'].unique()):
                plt.figure(figsize=(10, 6))
                cdf = df[df['case'] == case]
                fixed_n = [1, 2, 4, 8, 16]
                if 'variant' in cdf.columns:
                    for backend in sorted(cdf['variant'].unique()):
                        bdf = cdf[cdf['variant'] == backend]
                        bdf_agg = bdf.groupby('n_dim', as_index=False)['mean_time'].mean()
                        series = (bdf_agg.set_index('n_dim')['mean_time'] / 1000.0).reindex(fixed_n)
                        label = backend if backend != 'gemv_direct' else 'gemv-direct'
                        plt.plot(fixed_n, series.values, marker='o', linewidth=2, label=label)
                else:
                    cdf_agg = cdf.groupby('n_dim', as_index=False)['mean_time'].mean()
                    series = (cdf_agg.set_index('n_dim')['mean_time'] / 1000.0).reindex(fixed_n)
                    plt.plot(fixed_n, series.values, marker='o', linewidth=2, color='blue', label='auto')
                plt.xlabel('N Dimension')
                plt.ylabel('Mean Time (µs)')
                plt.title(f'Small-N GEMV Performance vs N (case: {case})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.ticklabel_format(style='plain', axis='y')
                plt.savefig(f'{output_dir}/smalln_performance_{case}.png', dpi=300, bbox_inches='tight')
                plt.close()

    # Additional visualization: Small-N per-backend charts for clear comparisons
    if smalln_analysis and 'data' in smalln_analysis:
        df = smalln_analysis['data']
        if len(df) > 0 and 'variant' in df.columns and 'n_dim' in df.columns:
            fixed_n = [1, 2, 4, 8, 16]
            for backend in sorted(df['variant'].unique()):
                plt.figure(figsize=(10, 6))
                bdf = df[df['variant'] == backend]
                bdf_agg = bdf.groupby('n_dim', as_index=False)['mean_time'].mean()
                series = (bdf_agg.set_index('n_dim')['mean_time'] / 1000.0).reindex(fixed_n)
                label = backend if backend != 'gemv_direct' else 'gemv-direct'
                plt.plot(fixed_n, series.values, marker='o', linewidth=2, label=label)
                plt.xticks(fixed_n)
                plt.xlabel('N Dimension')
                plt.ylabel('Mean Time (µs)')
                plt.title(f'Small-N GEMV Performance — Backend: {label}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.ticklabel_format(style='plain', axis='y')
                plt.savefig(f'{output_dir}/smalln_performance_backend_{label}.png', dpi=300, bbox_inches='tight')
                plt.close()

    # Combined visualization: Small-N per-backend grid (all together)
    if smalln_analysis and 'data' in smalln_analysis:
        df = smalln_analysis['data']
        if len(df) > 0 and 'variant' in df.columns and 'n_dim' in df.columns:
            fixed_n = [1, 2, 4, 8, 16]
            backends = sorted(df['variant'].unique())
            agg = df.groupby(['variant', 'n_dim'], as_index=False)['mean_time'].mean()
            # Determine global y-range for consistent scales
            y_vals = (agg['mean_time'] / 1000.0).values
            y_min = float(np.nanmin(y_vals)) if len(y_vals) else 0.0
            y_max = float(np.nanmax(y_vals)) if len(y_vals) else 1.0

            n_cols = 3
            n_rows = int(np.ceil(len(backends) / n_cols)) if len(backends) else 1
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), sharex=True, sharey=True)
            axes = np.array(axes).reshape(-1)

            for i, backend in enumerate(backends):
                ax = axes[i]
                bdf = agg[agg['variant'] == backend]
                series = (bdf.set_index('n_dim')['mean_time'] / 1000.0).reindex(fixed_n)
                label = backend if backend != 'gemv_direct' else 'gemv-direct'
                ax.plot(fixed_n, series.values, marker='o', linewidth=2)
                ax.set_title(label)
                ax.set_xticks(fixed_n)
                ax.grid(True, alpha=0.3)

            # Hide unused axes if any
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            # Apply consistent y-limits
            for ax in fig.get_axes():
                ax.set_ylim(y_min * 0.95, y_max * 1.05)

            fig.suptitle('Small-N GEMV Performance — Per Backend (aligned domain)', y=0.99)
            fig.text(0.5, 0.01, 'N Dimension', ha='center')
            fig.text(0.01, 0.5, 'Mean Time (µs)', va='center', rotation='vertical')
            fig.tight_layout(rect=[0.03, 0.03, 1, 0.95])
            plt.savefig(f'{output_dir}/smalln_performance_backends_grid.png', dpi=300, bbox_inches='tight')
            plt.close()

    # Separate visualization: Direct Small-N kernels comparison only
    if smalln_analysis and 'data' in smalln_analysis:
        df = smalln_analysis['data']
        if len(df) > 0 and 'variant' in df.columns and 'n_dim' in df.columns:
            direct_df = df[df['variant'].str.contains('gemv_direct', na=False)].copy()
            if len(direct_df) > 0:
                plt.figure(figsize=(10, 6))
                for v in sorted(direct_df['variant'].unique()):
                    vdf = direct_df[direct_df['variant'] == v].sort_values('n_dim')
                    plt.plot(vdf['n_dim'], vdf['mean_time'], marker='o', linewidth=2, label=v)
                plt.xlabel('N Dimension')
                plt.ylabel('Mean Time (ns)')
                plt.title('Direct Small-N GEMV Kernels (n1,n2,n4,n8,n16)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(f'{output_dir}/smalln_direct_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()

    # Separate visualization: Direct GEMM kernels comparison (MLX vs MPS) on matching shapes
    if gemm_analysis and 'data' in gemm_analysis:
        gdf = gemm_analysis['data']
        if len(gdf) > 0 and set(['variant', 'm_dim', 'k_dim', 'n_dim']).issubset(gdf.columns):
            plt.figure(figsize=(12, 7))
            for (variant, m_dim, k_dim), vdf in gdf.groupby(['variant', 'm_dim', 'k_dim']):
                vdf = vdf.sort_values('n_dim')
                label = f"{variant} {m_dim}x{k_dim}"
                plt.plot(vdf['n_dim'], vdf['mean_time'] / 1000.0, marker='o', linewidth=2, label=label)
            plt.xlabel('N Dimension')
            plt.ylabel('Mean Time (µs)')
            plt.title('Direct GEMM Kernels on matching shapes')
            plt.legend(ncol=2)
            plt.grid(True, alpha=0.3)
            plt.ticklabel_format(style='plain', axis='y')
            plt.savefig(f'{output_dir}/gemm_direct_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

    # Overlay visualization (split): Direct vs Dispatcher per (m_dim, k_dim) and backend
    if gemm_analysis and gemm_disp_analysis and 'data' in gemm_analysis and 'data' in gemm_disp_analysis:
        dg = gemm_analysis['data']
        dd = gemm_disp_analysis['data']
        if len(dg) > 0 and len(dd) > 0:
            # Map direct variants to dispatcher counterparts
            variant_pairs = {
                'gemm_direct_mlx': 'mlx',
                'gemm_direct_mps': 'mps',
                'gemm_direct_tiled': 'gemm_tiled',
            }

            # Iterate per (m_dim, k_dim) to produce focused comparisons
            for (m_dim, k_dim), dg_family in dg.groupby(['m_dim', 'k_dim']):
                dd_family = dd[(dd['m_dim'] == m_dim) & (dd['k_dim'] == k_dim)]
                if len(dd_family) == 0:
                    continue

                # Build a unified N domain to align line segments
                n_domain = sorted(set(dg_family['n_dim'].tolist() + dd_family['n_dim'].tolist()))

                plt.figure(figsize=(10, 6))
                plotted_any = False
                for direct_var, disp_var in variant_pairs.items():
                    d_line = dg_family[dg_family['variant'] == direct_var]
                    v_line = dd_family[dd_family['variant'] == disp_var]
                    if len(d_line) == 0 and len(v_line) == 0:
                        continue

                    # Reindex both to the unified N domain for consistent x-axis and gaps
                    d_series = (d_line.set_index('n_dim')['mean_time'] / 1000.0).reindex(n_domain)
                    v_series = (v_line.set_index('n_dim')['mean_time'] / 1000.0).reindex(n_domain)

                    plt.plot(n_domain, d_series.values, marker='o', linewidth=2, label=f"{direct_var.split('_')[-1]} direct")
                    plt.plot(n_domain, v_series.values, marker='o', linewidth=2, label=f"{disp_var} dispatcher")
                    plotted_any = True

                if not plotted_any:
                    plt.close()
                    continue

                plt.xlabel('N Dimension')
                plt.ylabel('Mean Time (µs)')
                plt.title(f'GEMM Direct vs Dispatcher — {m_dim}x{k_dim}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.ticklabel_format(style='plain', axis='y')
                plt.savefig(f'{output_dir}/gemm_overlay_{m_dim}x{k_dim}.png', dpi=300, bbox_inches='tight')
                plt.close()

    print(f"\nVisualizations saved to '{output_dir}' directory")

def analyze_gemm_direct(df):
    """Analyze direct GEMM (non-small-N) kernels from direct benches.

    Returns a dict containing aggregated mean_time per (variant, m_dim, k_dim, n_dim).
    """
    import pandas as pd
    gemm_df = df[df['benchmark'].str.contains('gemm_kernels_direct', case=False, na=False)].copy()
    # Focus analysis on f16 runs
    gemm_df = gemm_df[gemm_df['benchmark'].str.contains('f16', case=False, na=False)]
    if len(gemm_df) == 0:
        return {'data': pd.DataFrame(), 'variants': []}

    # Extract dimensions from parameter labels like 'MxKxN_a1_b0'
    dims = gemm_df['parameters'].str.extract(r'(?P<m_dim>\d+)x(?P<k_dim>\d+)x(?P<n_dim>\d+)')
    gemm_df = gemm_df.join(dims)
    for col in ['m_dim', 'k_dim', 'n_dim']:
        gemm_df[col] = pd.to_numeric(gemm_df[col], errors='coerce')
    gemm_df = gemm_df.dropna(subset=['m_dim', 'k_dim', 'n_dim'])

    # Normalize variant names to consistent tokens
    def norm_variant(v):
        lv = v.lower() if isinstance(v, str) else ''
        if 'gemm_direct_mlx' in lv:
            return 'gemm_direct_mlx'
        if 'gemm_direct_mps' in lv:
            return 'gemm_direct_mps'
        if 'gemm_direct_tiled' in lv:
            return 'gemm_direct_tiled'
        return lv

    gemm_df['variant'] = gemm_df['variant'].apply(norm_variant)

    agg = gemm_df.groupby(['variant', 'm_dim', 'k_dim', 'n_dim'], as_index=False)['mean_time'].mean()

    best_per_shape = []
    transitions = {}
    if len(agg) > 0:
        print("\n[GEMM Direct] Best backend per (M,K,N):")
        for (m_dim, k_dim), grp in agg.groupby(['m_dim', 'k_dim']):
            best = (
                grp.sort_values('mean_time')
                .drop_duplicates('n_dim', keep='first')
                .sort_values('n_dim')
            )
            for _, row in best.iterrows():
                best_per_shape.append({
                    'm_dim': int(row['m_dim']),
                    'k_dim': int(row['k_dim']),
                    'n_dim': int(row['n_dim']),
                    'variant': row['variant'],
                    'mean_time': row['mean_time'],
                })
            trans = []
            prev_variant = None
            for _, row in best.iterrows():
                variant = row['variant']
                n_dim = int(row['n_dim'])
                if variant != prev_variant:
                    trans.append({'start': n_dim, 'variant': variant})
                    prev_variant = variant
            if trans:
                transitions[(int(m_dim), int(k_dim))] = trans
                segments = []
                for idx, entry in enumerate(trans):
                    start = entry['start']
                    variant = entry['variant']
                    if idx + 1 < len(trans):
                        end = trans[idx + 1]['start'] - 1
                        segments.append(f"{variant}: N ∈ [{start}, {end}]")
                    else:
                        segments.append(f"{variant}: N ≥ {start}")
                print(f"  {m_dim}x{k_dim}: " + "; ".join(segments))

    return {
        'data': agg,
        'variants': sorted(agg['variant'].unique()),
        'best_per_shape': best_per_shape,
        'transitions': transitions,
    }

def analyze_gemm_dispatcher(df):
    """Analyze dispatcher GEMM benchmarks with matching shapes.

    Returns aggregated mean_time per (variant, m_dim, k_dim, n_dim).
    """
    import pandas as pd
    disp_df = df[df['benchmark'].str.contains('matmul_dispatcher_gemm', case=False, na=False)].copy()
    # Focus on f16
    disp_df = disp_df[disp_df['benchmark'].str.contains('f16', case=False, na=False)]
    if len(disp_df) == 0:
        return {'data': pd.DataFrame(), 'variants': []}

    dims = disp_df['parameters'].str.extract(r'(?P<m_dim>\d+)x(?P<k_dim>\d+)x(?P<n_dim>\d+)')
    disp_df = disp_df.join(dims)
    for col in ['m_dim', 'k_dim', 'n_dim']:
        disp_df[col] = pd.to_numeric(disp_df[col], errors='coerce')
    disp_df = disp_df.dropna(subset=['m_dim', 'k_dim', 'n_dim'])

    # Normalize variant names
    def norm(v):
        lv = v.lower() if isinstance(v, str) else ''
        if lv in ['mlx', 'mps', 'gemv', 'gemm_tiled', 'auto']:
            return lv
        return lv
    disp_df['variant'] = disp_df['variant'].apply(norm)

    agg = disp_df.groupby(['variant', 'm_dim', 'k_dim', 'n_dim'], as_index=False)['mean_time'].mean()
    agg = agg[agg['variant'] != 'noop']

    best_per_shape = []
    transitions = {}
    if len(agg) > 0:
        print("\n[GEMM Dispatcher] Best backend per (M,K,N):")
        for (m_dim, k_dim), grp in agg.groupby(['m_dim', 'k_dim']):
            best = (
                grp.sort_values('mean_time')
                .drop_duplicates('n_dim', keep='first')
                .sort_values('n_dim')
            )
            for _, row in best.iterrows():
                best_per_shape.append({
                    'm_dim': int(row['m_dim']),
                    'k_dim': int(row['k_dim']),
                    'n_dim': int(row['n_dim']),
                    'variant': row['variant'],
                    'mean_time': row['mean_time'],
                })
            trans = []
            prev_variant = None
            for _, row in best.iterrows():
                variant = row['variant']
                n_dim = int(row['n_dim'])
                if variant != prev_variant:
                    trans.append({'start': n_dim, 'variant': variant})
                    prev_variant = variant
            if trans:
                transitions[(int(m_dim), int(k_dim))] = trans
                segments = []
                for idx, entry in enumerate(trans):
                    start = entry['start']
                    variant = entry['variant']
                    if idx + 1 < len(trans):
                        end = trans[idx + 1]['start'] - 1
                        segments.append(f"{variant}: N ∈ [{start}, {end}]")
                    else:
                        segments.append(f"{variant}: N ≥ {start}")
                print(f"  {m_dim}x{k_dim}: " + "; ".join(segments))

    return {
        'data': agg,
        'variants': sorted(agg['variant'].unique()),
        'best_per_shape': best_per_shape,
        'transitions': transitions,
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze benchmark results for optimal kernel crossover points')
    parser.add_argument('csv_path', help='Path to benchmark results CSV file')
    parser.add_argument('--output-dir', default='benchmark_analysis', help='Output directory for visualizations')

    args = parser.parse_args()

    # Load benchmark data
    df = load_benchmark_data(args.csv_path)
    if df is None:
        sys.exit(1)

    # Analyze crossovers
    softmax_analysis = analyze_softmax_crossover(df)
    smalln_analysis = analyze_smalln_gemv_crossover(df)
    gemm_analysis = analyze_gemm_direct(df)
    gemm_disp_analysis = analyze_gemm_dispatcher(df)

    # Generate recommendations
    recommendations = generate_recommendations(softmax_analysis, smalln_analysis)

    # Create visualizations
    create_visualization(softmax_analysis, smalln_analysis, gemm_analysis, gemm_disp_analysis, output_dir=args.output_dir)

    print(f"\nAnalysis complete! Check '{args.output_dir}' for visualizations.")
    return recommendations

if __name__ == "__main__":
    main()
