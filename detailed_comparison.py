#!/usr/bin/env python3

import json
import sys
import statistics

def analyze_detailed(filename):
    events = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                events.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    # Get all timing data
    op_times = {}
    for event in events:
        if event['event']['type'] == 'InternalKernelCompleted':
            data = event['event']['data']
            op_name = data.get('internal_kernel_name', data.get('op_name', 'unknown'))
            duration = data['duration_us']
            if op_name not in op_times:
                op_times[op_name] = []
            op_times[op_name].append(duration / 1000)  # Convert to ms

    return {
        'total_events': len(events),
        'op_statistics': {op: {
            'count': len(times),
            'total_ms': sum(times),
            'avg_ms': statistics.mean(times) if times else 0,
            'min_ms': min(times) if times else 0,
            'max_ms': max(times) if times else 0
        } for op, times in op_times.items()}
    }

def compare_graph_vs_legacy():
    legacy = analyze_detailed('legacy-noprof.jsonl')
    graph = analyze_detailed('graph-noprof.jsonl')

    print("DETAILED GRAPH vs LEGACY COMPARISON (non-profiling)")
    print("=" * 60)
    print(f"Legacy events: {legacy['total_events']:,}")
    print(f"Graph events:  {graph['total_events']:,}")
    print(f"Event difference: {graph['total_events'] - legacy['total_events']:,} ({(graph['total_events']/legacy['total_events'] - 1) * 100:.1f}%)")
    print()

    # Compare operation statistics
    print("OPERATION BREAKDOWN:")
    print("-" * 60)

    all_ops = set(legacy['op_statistics'].keys()) | set(graph['op_statistics'].keys())

    # Sort by total time in legacy
    sorted_ops = sorted(all_ops, key=lambda op: legacy['op_statistics'].get(op, {}).get('total_ms', 0), reverse=True)

    for op in sorted_ops:
        legacy_stats = legacy['op_statistics'].get(op, {})
        graph_stats = graph['op_statistics'].get(op, {})

        if legacy_stats:
            legacy_total = legacy_stats['total_ms']
            legacy_count = legacy_stats['count']
            legacy_avg = legacy_stats['avg_ms']
        else:
            legacy_total = legacy_count = legacy_avg = 0

        if graph_stats:
            graph_total = graph_stats['total_ms']
            graph_count = graph_stats['count']
            graph_avg = graph_stats['avg_ms']
        else:
            graph_total = graph_count = graph_avg = 0

        if legacy_total > 0 or graph_total > 0:
            diff_ms = graph_total - legacy_total
            diff_pct = (graph_total / legacy_total - 1) * 100 if legacy_total > 0 else float('inf')

            print(f"{op}:")
            print(f"  Legacy: {legacy_count:>5} ops, {legacy_total:>8.1f}ms total, {legacy_avg:>6.3f}ms avg")
            print(f"  Graph:  {graph_count:>5} ops, {graph_total:>8.1f}ms total, {graph_avg:>6.3f}ms avg")
            print(f"  Delta:  {graph_count - legacy_count:>+5} ops, {diff_ms:>+8.1f}ms total, {diff_pct:+6.1f}% change")
            print()

    # Total timing comparison
    legacy_total_time = sum(stats['total_ms'] for stats in legacy['op_statistics'].values())
    graph_total_time = sum(stats['total_ms'] for stats in graph['op_statistics'].values())

    print("TOTAL INTERNAL KERNEL TIME:")
    print("-" * 60)
    print(f"Legacy total: {legacy_total_time:.1f}ms")
    print(f"Graph total:  {graph_total_time:.1f}ms")
    print(f"Difference:   {graph_total_time - legacy_total_time:.1f}ms ({(graph_total_time/legacy_total_time - 1) * 100:.1f}%)")
    print()

    print("TOP TIME-CONSUMING OPERATIONS:")
    print("-" * 60)
    for op in sorted(all_ops, key=lambda op: max(graph['op_statistics'].get(op, {}).get('total_ms', 0), legacy['op_statistics'].get(op, {}).get('total_ms', 0)), reverse=True)[:10]:
        legacy_total = legacy['op_statistics'].get(op, {}).get('total_ms', 0)
        graph_total = graph['op_statistics'].get(op, {}).get('total_ms', 0)
        print(f"{op}: Legacy={legacy_total:.1f}ms, Graph={graph_total:.1f}ms")

if __name__ == "__main__":
    compare_graph_vs_legacy()
