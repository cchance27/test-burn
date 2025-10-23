#!/usr/bin/env python3

import json
import sys
import statistics
from datetime import datetime

def analyze_total_timing(filename):
    print(f"\n=== Total timing analysis for {filename} ===")

    events = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                events.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    # Find first and last timestamps
    timestamps = []
    for event in events:
        ts_str = event.get('timestamp')
        if ts_str:
            # Parse ISO timestamp
            try:
                dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                timestamps.append(dt.timestamp() * 1000000)  # microseconds
            except:
                continue

    if not timestamps:
        print("No valid timestamps found")
        return

    min_ts = min(timestamps)
    max_ts = max(timestamps)
    total_duration_us = max_ts - min_ts
    total_duration_ms = total_duration_us / 1000

    print(f"Total trace duration: {total_duration_ms:.1f}ms")
    # Look for ForwardStep events which might be the complete generation cycles
    forward_steps = []
    step_times = []
    for event in events:
        if event['event']['type'] == 'ForwardStep':
            forward_steps.append(event)

    print(f"ForwardStep events: {len(forward_steps)}")
    if forward_steps:
        # If there are ForwardStep events, let's calculate timing between them
        prev_ts = None
        for event in forward_steps:
            ts_str = event.get('timestamp')
            if ts_str:
                try:
                    dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    current_ts = dt.timestamp() * 1000000
                    if prev_ts is not None:
                        step_times.append(current_ts - prev_ts)
                    prev_ts = current_ts
                except:
                    continue

        if step_times:
            avg_step_ms = statistics.mean(step_times) / 1000
            print(f"Average ForwardStep interval: {avg_step_ms:.3f}ms")
            for i, step_time in enumerate(step_times[:5]):  # Show first 5
                print(f"  Step {i+1}: {step_time/1000:.1f}ms")
            if len(step_times) > 5:
                print("    ... and more steps")

    # Count generation loops vs other operations
    sync_ops = [e for e in events if 'sync' in e['event']['data'].get('internal_kernel_name', '')]
    gen_loop_ops = [e for e in events if e['event']['type'] == 'InternalKernelCompleted' and e['event']['data'].get('parent_op_name') == 'generation_loop']

    print(f"\nSync operations: {len(sync_ops)}")
    print(f"Generation loop operations: {len(gen_loop_ops)}")

    if forward_steps and step_times:
        num_steps = len(forward_steps) - 1  # intervals between events
        per_step_sync = len(sync_ops) / num_steps
        per_step_gen = len(gen_loop_ops) / num_steps
        print(f"\nPer step averages (across {num_steps} steps):")
        print(f"  Sync ops per step: {per_step_sync:.1f}")
        print(f"  Gen loop ops per step: {per_step_gen:.1f}")
        print(f"  Average step time: {statistics.mean(step_times) / 1000:.3f}ms")
def main():
    files = ['legacy-noprof.jsonl', 'graph-noprof.jsonl']

    for f in files:
        try:
            analyze_total_timing(f)
        except FileNotFoundError:
            print(f"{f} not found")

if __name__ == "__main__":
    main()
