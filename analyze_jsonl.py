#!/usr/bin/env python3

import json
import sys
from collections import defaultdict
import statistics

def analyze_file(filename):
    print(f"\n=== Analyzing {filename} ===")

    # Read all events
    events = []
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f):
            try:
                events.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue

    # Group events by type
    event_counts = defaultdict(int)
    duration_totals = defaultdict(float)
    duration_counts = defaultdict(int)
    generation_durations = []
    prompt_durations = []

    for event in events:
        event_type = event['event']['type']
        event_counts[event_type] += 1

        data = event['event'].get('data', {})

        # Track durations for operation types
        if 'duration_us' in data:
            duration_totals[event_type] += data['duration_us']
            duration_counts[event_type] += 1

    # Look for generation loop and prompt processing timings
    internal_kernel_durations = defaultdict(list)

    for event in events:
        if event['event']['type'] == 'InternalKernelCompleted':
            data = event['event']['data']
            operation_name = data.get('parent_op_name', '')
            if operation_name in ['generation_loop', 'prompt_processing']:
                internal_kernel_durations[operation_name].append(data['duration_us'])

    print(f"Total events: {len(events)}")
    print(f"Event type counts:")
    for event_type, count in sorted(event_counts.items(), key=lambda x: x[1], reverse=True):
        duration_str = ""
        if event_type in duration_totals:
            total_ms = duration_totals[event_type] / 1000
            avg_ms = (duration_totals[event_type] / duration_counts[event_type]) / 1000
            duration_str = f" | Total: {total_ms:.1f}ms | Avg: {avg_ms:.3f}ms"
        print(f"  {event_type}: {count}{duration_str}")

    print(f"\nInternal kernel summaries:")
    for op_name, durations in internal_kernel_durations.items():
        total_ms = sum(durations) / 1000
        avg_ms = statistics.mean(durations) / 1000 if durations else 0
        min_ms = min(durations) / 1000 if durations else 0
        max_ms = max(durations) / 1000 if durations else 0
        count = len(durations)
        print(f"  {op_name}: {count} ops | Total: {total_ms:.1f}ms | Avg: {avg_ms:.1f}ms | Min: {min_ms:.1f}ms | Max: {max_ms:.1f}ms")

    # Specifically analyze sync operations
    sync_events = [e for e in events if 'sync' in e['event']['data'].get('internal_kernel_name', '') or 'sync' in e['event']['data'].get('op_name', '')]
    print(f"\nSync operations: {len(sync_events)}")
    if sync_events:
        sync_times = []
        for e in sync_events:
            data = e['event']['data']
            duration = data.get('duration_us', 0)
            sync_times.append(duration / 1000)
        sync_total = sum(sync_times)
        sync_avg = statistics.mean(sync_times) if sync_times else 0
        print(f"  Total sync time: {sync_total:.1f}ms")
        print(f"  Average sync time: {sync_avg:.1f}ms")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_jsonl.py <filename>")
        sys.exit(1)

    analyze_file(sys.argv[1])
