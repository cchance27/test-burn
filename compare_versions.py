#!/usr/bin/env python3

import json
import sys
from collections import defaultdict
import statistics

def analyze_file(filename):
    events = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                events.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    # Get generation loop stats
    # In prof builds: look for iteration_total events (real timings)
    # In noprof builds: this data doesn't exist, so use the sync times as approximation
    gen_loop_times = []
    for event in events:
        if event['event']['type'] == 'InternalKernelCompleted':
            data = event['event']['data']
            if data.get('internal_kernel_name') == 'iteration_total':
                duration = data['duration_us']
                if duration > 0:  # Only include real timing data
                    gen_loop_times.append(duration)
            elif data.get('parent_op_name') == 'generation_loop' and not gen_loop_times:  # fallback
                duration = data['duration_us']
                if duration > 0:
                    gen_loop_times.append(duration)

    # Sync operations
    sync_events = [e for e in events if 'sync' in e['event']['data'].get('internal_kernel_name', '')]
    sync_times = [e['event']['data']['duration_us'] for e in sync_events]

    # Per generation step averages
    total_gen_steps = len(gen_loop_times)
    avg_gen_us = statistics.mean(gen_loop_times) if gen_loop_times else 0
    avg_sync_us = statistics.mean(sync_times) if sync_times else 0

    # Use only generation loop time for total - sync time affects throughput indirectly
    total_time_per_step_us = avg_gen_us



    return {
        'filename': filename,
        'gen_steps': total_gen_steps,
        'avg_gen_ms': avg_gen_us / 1000,
        'avg_sync_ms': avg_sync_us / 1000,
        'total_per_step_ms': total_time_per_step_us / 1000,
        'sync_count': len(sync_events),
        'total_sync_ms': sum(sync_times) / 1000 if sync_times else 0,
        'total_gen_ms': sum(gen_loop_times) / 1000 if gen_loop_times else 0
    }

def main():
    files = ['legacy-noprof.jsonl', 'graph-noprof.jsonl', 'legacy.jsonl', 'graph.jsonl']

    print("Performance Analysis of Graph vs Legacy Inference")
    print("=" * 60)

    results = []
    for f in files:
        try:
            result = analyze_file(f)
            results.append(result)
        except FileNotFoundError:
            print(f"Warning: {f} not found")
            continue

    for r in results:
        print(f"\n{r['filename']}:")
        print(f"  Generation steps: {r['gen_steps']}")
        print(f"  Average gen time per step: {r['avg_gen_ms']:.3f}ms")
        print(f"  Average sync time per step: {r['avg_sync_ms']:.3f}ms")
        print(f"  Total per step: {r['total_per_step_ms']:.3f}ms")

    if len(results) >= 2:
        # Compare the key ones
        legacy_prof = next((r for r in results if r['filename'] == 'legacy-noprof.jsonl'), None)
        graph_prof = next((r for r in results if r['filename'] == 'graph-noprof.jsonl'), None)

        if legacy_prof and graph_prof:
            print(f"\n{'=' * 60}")
            print("Key Comparison (non-profiling versions):")
            print(f"Legacy total per step:  {legacy_prof['total_per_step_ms']:.3f}ms")
            print(f"Graph total per step:   {graph_prof['total_per_step_ms']:.3f}ms")
            print(f"Difference:            {graph_prof['total_per_step_ms'] - legacy_prof['total_per_step_ms']:.3f}ms ({(graph_prof['total_per_step_ms']/legacy_prof['total_per_step_ms'] - 1) * 100:.1f}%)")
            print()
            print("Breakdown of difference:")
            print(f"  Generation time delta: {graph_prof['avg_gen_ms'] - legacy_prof['avg_gen_ms']:.3f}ms")
            print(f"  Sync time delta:       {graph_prof['avg_sync_ms'] - legacy_prof['avg_sync_ms']:.3f}ms")

if __name__ == "__main__":
    main()
