#!/usr/bin/env python3

import argparse
import json
import math
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PERCENTILES = (50, 90, 95, 99)


def percentile(sorted_samples: List[float], pct: float) -> float:
    if not sorted_samples:
        return 0.0
    rank = (pct / 100.0) * (len(sorted_samples) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_samples[lower]
    weight = rank - lower
    return sorted_samples[lower] * (1 - weight) + sorted_samples[upper] * weight


def summarize(samples: Iterable[int]) -> Dict[str, float]:
    values = list(samples)
    if not values:
        return {
            "count": 0,
            "total_ms": 0.0,
            "avg_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }
    sorted_vals = sorted(values)
    total_us = sum(sorted_vals)
    total_ms = total_us / 1000.0
    avg_ms = total_ms / len(sorted_vals)
    min_ms = sorted_vals[0] / 1000.0
    max_ms = sorted_vals[-1] / 1000.0
    percentiles = {
        f"p{pct}_ms": percentile(sorted_vals, pct) / 1000.0 for pct in PERCENTILES
    }
    return {
        "count": len(sorted_vals),
        "total_ms": total_ms,
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        **percentiles,
    }


def render_summary(title: str, stats: Dict[str, float], indent: int = 2) -> str:
    pad = " " * indent
    if stats["count"] == 0:
        return f"{pad}{title}: no samples"
    percentile_parts = ", ".join(
        f"{key}={value:.3f}ms"
        for key, value in stats.items()
        if key.startswith("p")
    )
    return (
        f"{pad}{title}: count={int(stats['count'])} | total={stats['total_ms']:.2f}ms | "
        f"avg={stats['avg_ms']:.3f}ms | min={stats['min_ms']:.3f}ms | "
        f"max={stats['max_ms']:.3f}ms | {percentile_parts}"
    )


def analyze_file(filename: str, top_n: int, kernel_top: int, include_kernel_totals: bool, kernel_filter: str) -> None:
    path = Path(filename)
    print(f"\n=== Analyzing {path} ===")

    event_counts: Counter[str] = Counter()
    event_duration_samples: Dict[str, List[int]] = defaultdict(list)

    internal_parent_samples: Dict[str, List[int]] = defaultdict(list)
    internal_kernel_samples: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    kernel_samples: Dict[str, List[int]] = defaultdict(list)
    gpu_wait_samples: Dict[str, List[int]] = defaultdict(list)
    gpu_kernel_samples: Dict[str, List[int]] = defaultdict(list)
    gpu_kernel_base_samples: Dict[str, List[int]] = defaultdict(list)
    sync_samples: List[int] = []

    total_events = 0
    with path.open("r") as handle:
        for line_num, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Error parsing line {line_num}: {exc}")
                continue

            event = payload.get("event", {})
            event_type = event.get("type")
            if not event_type:
                continue
            data = event.get("data", {})

            event_counts[event_type] += 1
            total_events += 1

            duration_us = data.get("duration_us")
            if isinstance(duration_us, (int, float)):
                duration_int = int(duration_us)
                event_duration_samples[event_type].append(duration_int)

                kernel_name = data.get("internal_kernel_name", "") or data.get("op_name", "")
                if "sync" in kernel_name.lower():
                    sync_samples.append(duration_int)

            if event_type == "InternalKernelCompleted" and duration_us is not None:
                parent = data.get("parent_op_name", "<unknown>")
                kernel = data.get("internal_kernel_name", "<unknown>")
                duration_int = int(duration_us)
                internal_parent_samples[parent].append(duration_int)
                internal_kernel_samples[(parent, kernel)].append(duration_int)
                kernel_samples[kernel].append(duration_int)

            if event_type == "GpuOpCompleted" and duration_us is not None:
                op_name = data.get("op_name", "<unknown>")
                duration_int = int(duration_us)
                if "cb_wait" in op_name or op_name.endswith("wait"):
                    gpu_wait_samples[op_name].append(duration_int)
                else:
                    gpu_kernel_samples[op_name].append(duration_int)
                    base_name = op_name.split("/")[-1] if op_name else "<unknown>"
                    if base_name.startswith("sdpa_block_") and base_name.endswith("_op"):
                        base_name = "sdpa"
                    elif base_name.startswith("mlp_swiglu_block_") and base_name.endswith("_op"):
                        base_name = "mlp_swiglu"
                    elif base_name.endswith("_op"):
                        base_name = base_name.rsplit("_", 1)[0]
                    base_name = base_name.rstrip("_")
                    gpu_kernel_base_samples[base_name].append(duration_int)

    print(f"Total events: {total_events}")
    print("\nEvent type summary:")
    for event_type, count in event_counts.most_common():
        stats = summarize(event_duration_samples[event_type])
        print(render_summary(f"{event_type} (count={count})", stats, indent=2))

    if internal_parent_samples:
        print("\nInternal kernel by parent:")
        parent_totals = sorted(
            (
                parent,
                summarize(samples),
            )
            for parent, samples in internal_parent_samples.items()
        )
        parent_totals.sort(key=lambda item: item[1]["total_ms"], reverse=True)
        for parent, stats in parent_totals[:top_n]:
            print(render_summary(parent, stats, indent=2))

        print("\nInternal kernel by parent/op:")
        kernel_totals = []
        for (parent, kernel), samples in internal_kernel_samples.items():
            stats = summarize(samples)
            kernel_totals.append(((parent, kernel), stats))
        kernel_totals.sort(key=lambda item: item[1]["total_ms"], reverse=True)
        for (parent, kernel), stats in kernel_totals[:top_n]:
            title = f"{parent} :: {kernel}"
            print(render_summary(title, stats, indent=2))

    if kernel_samples:
        print("\nKernel aggregate (all parents):")
        kernel_totals = []
        for kernel, samples in kernel_samples.items():
            if not include_kernel_totals and (
                kernel.endswith("_total")
                or kernel.endswith("cb_wait")
                or kernel in {"iteration_total", "forward_step_total"}
            ):
                continue
            if kernel_filter and kernel_filter.lower() not in kernel.lower():
                continue
            kernel_totals.append((kernel, summarize(samples)))

        if not kernel_totals:
            print("  (no kernels matched)")
        else:
            kernel_totals.sort(key=lambda item: item[1]["total_ms"], reverse=True)
            for kernel, stats in kernel_totals[:kernel_top]:
                print(render_summary(kernel, stats, indent=2))

    if gpu_kernel_samples:
        print("\nGPU kernel durations (excluding waits):")
        kernel_totals = []
        for op_name, samples in gpu_kernel_samples.items():
            if kernel_filter and kernel_filter.lower() not in op_name.lower():
                continue
            kernel_totals.append((op_name, summarize(samples)))

        if not kernel_totals:
            print("  (no kernels matched)")
        else:
            kernel_totals.sort(key=lambda item: item[1]["total_ms"], reverse=True)
            for op_name, stats in kernel_totals[:top_n]:
                print(render_summary(op_name, stats, indent=2))

    if gpu_kernel_base_samples:
        print("\nGPU kernel aggregate (by kernel type):")
        base_totals = []
        for base_name, samples in gpu_kernel_base_samples.items():
            if kernel_filter and kernel_filter.lower() not in base_name.lower():
                continue;
            base_totals.append((base_name, summarize(samples)))
        if not base_totals:
            print("  (no kernels matched)")
        else:
            base_totals.sort(key=lambda item: item[1]["total_ms"], reverse=True)
            for base_name, stats in base_totals[:top_n]:
                print(render_summary(base_name, stats, indent=2))

    if gpu_wait_samples:
        print("\nGPU wait breakdown:")
        gpu_totals = [
            (op_name, summarize(samples)) for op_name, samples in gpu_wait_samples.items()
        ]
        gpu_totals.sort(key=lambda item: item[1]["total_ms"], reverse=True)
        for op_name, stats in gpu_totals[:top_n]:
            print(render_summary(op_name, stats, indent=2))

    print("\nSync operations:")
    print(render_summary("sync kernels", summarize(sync_samples), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize metallic instrumentation JSONL output.")
    parser.add_argument("filename", help="Path to the JSONL file to analyze.")
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top entries to display for kernel and GPU wait breakdowns (default: 20).",
    )
    parser.add_argument(
        "--kernel-top",
        type=int,
        default=50,
        help="Number of kernels to show in the aggregate kernel view (default: 50).",
    )
    parser.add_argument(
        "--include-kernel-totals",
        action="store_true",
        help="Include *_total and cb_wait synthetic kernels in the aggregate kernel view.",
    )
    parser.add_argument(
        "--kernel-filter",
        default="",
        help="Optional substring filter for kernel names (case-insensitive).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_file(
        args.filename,
        args.top,
        args.kernel_top,
        args.include_kernel_totals,
        args.kernel_filter,
    )
