#!/usr/bin/env python3

import argparse
import json
import math
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PERCENTILES = (95, 99)
MATMUL_BASE_KEYS = ("op", "batch", "m", "n", "k", "tA", "tB")
MATMUL_SCOPE_KEYS = MATMUL_BASE_KEYS + ("backend",)


def parse_matmul_scope(op_name: str, data: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Parse matmul metadata from either the data field (new) or op_name path (legacy)."""
    if "/cb_wait" in op_name or "/cb_gpu" in op_name:
        return None

    # New format: matmul parameters are in a nested data field
    if data is not None:
        # Check if there's a nested 'data' field with matmul parameters
        nested_data = data.get("data")
        if nested_data is not None and isinstance(nested_data, dict) and "op" in nested_data:
            return nested_data

    # Fallback: check if the top-level data has matmul parameters (direct format)
    if data is not None and "op" in data and "backend" in data:
        return data

    # Legacy format: parse from the op_name path
    segment = None
    for marker in ("matmul_cache/", "matmul/"):
        marker_idx = op_name.find(marker)
        if marker_idx != -1:
            segment = op_name[marker_idx:]
            break

    if segment is None:
        return None

    segments = segment.split("/")
    meta: Dict[str, str] = {}
    for seg in segments[1:]:
        if "::" in seg:
            break
        token = seg.split("#", 1)[0]
        if not token or "=" not in token:
            continue
        key, value = token.split("=", 1)
        meta[key] = value

    if not meta:
        return None
    return meta


def matmul_shape_key(meta: Dict[str, str]) -> str:
    keys = list(MATMUL_BASE_KEYS)
    parts = [f"{key}={meta.get(key, '?')}" for key in keys]
    extras = sorted(
        (key, value)
        for key, value in meta.items()
        if key not in MATMUL_SCOPE_KEYS
    )
    if extras:
        parts.extend(f"{key}={value}" for key, value in extras)
    return " | ".join(parts)


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
    filter_lower = kernel_filter.lower()

    event_counts: Counter[str] = Counter()
    event_duration_samples: Dict[str, List[int]] = defaultdict(list)
    backend_selection_by_backend: Counter[str] = Counter()
    backend_selection_by_reason: Counter[str] = Counter()
    backend_selection_by_op: Counter[str] = Counter()
    backend_selection_by_op_backend: Counter[Tuple[str, str]] = Counter()

    internal_parent_samples: Dict[str, List[int]] = defaultdict(list)
    internal_kernel_samples: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    kernel_samples: Dict[str, List[int]] = defaultdict(list)
    gpu_wait_samples: Dict[str, List[int]] = defaultdict(list)
    cb_gpu_samples: Dict[str, List[int]] = defaultdict(list)
    gpu_kernel_samples: Dict[str, List[int]] = defaultdict(list)
    gpu_kernel_base_samples: Dict[str, List[int]] = defaultdict(list)
    matmul_shape_samples: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    matmul_backend_samples: Dict[str, List[int]] = defaultdict(list)
    matmul_variant_samples: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    sync_samples: List[int] = []
    foundry_capture_kernel_counts: Counter[str] = Counter()
    foundry_capture_dispatches: List[int] = []

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
                if "/cb_gpu" in op_name:
                    cb_gpu_samples[op_name].append(duration_int)
                    continue
                if "cb_wait" in op_name or op_name.endswith("wait"):
                    gpu_wait_samples[op_name].append(duration_int)
                else:
                    gpu_kernel_samples[op_name].append(duration_int)
                    # Foundry capture aggregates kernel counts in a nested data map.
                    if op_name.startswith("foundry_capture#"):
                        capture_data = data.get("data")
                        if isinstance(capture_data, dict):
                            for key, value in capture_data.items():
                                if key == "dispatches":
                                    try:
                                        foundry_capture_dispatches.append(int(value))
                                    except (TypeError, ValueError):
                                        continue
                                    continue
                                if not key.startswith("k"):
                                    continue
                                if not isinstance(value, str):
                                    continue
                                if ":" not in value:
                                    continue
                                name, count_str = value.rsplit(":", 1)
                                try:
                                    count = int(count_str)
                                except ValueError:
                                    continue
                                foundry_capture_kernel_counts[name] += count
                    # Extract the optional data field for structured metadata
                    event_data = data.get("data")
                    matmul_meta = parse_matmul_scope(op_name, event_data)
                    if matmul_meta:
                        backend = matmul_meta.get("backend", "<unknown>")
                        base_shape = matmul_shape_key(matmul_meta)
                        matmul_shape_samples[base_shape][backend].append(duration_int)
                        matmul_backend_samples[backend].append(duration_int)
                        op_variant = matmul_meta.get("op", "<unknown>")
                        matmul_variant_samples[(op_variant, backend)].append(duration_int)
                    base_name = op_name.split("/")[-1] if op_name else "<unknown>"
                    if base_name.startswith("sdpa_block_") and base_name.endswith("_op"):
                        base_name = "sdpa"
                    elif base_name.startswith("mlp_swiglu_block_") and base_name.endswith("_op"):
                        base_name = "mlp_swiglu"
                    elif base_name.startswith("forward_cpu_block_"):
                        base_name = "forward_cpu"
                    elif base_name.endswith("_op"):
                        base_name = base_name.rsplit("_", 1)[0]
                    base_name = base_name.rstrip("_")
                    gpu_kernel_base_samples[base_name].append(duration_int)

            if event_type == "KernelBackendSelected":
                op_name = data.get("op_name", "<unknown>")
                backend = data.get("backend", "<unknown>")
                reason = data.get("reason", "<unknown>")

                backend_selection_by_backend[backend] += 1
                backend_selection_by_reason[reason] += 1
                backend_selection_by_op[op_name] += 1
                backend_selection_by_op_backend[(op_name, backend)] += 1

    print(f"Total events: {total_events}")
    print("\nEvent type summary:")
    for event_type, count in event_counts.most_common():
        stats = summarize(event_duration_samples[event_type])
        print(render_summary(f"{event_type} (count={count})", stats, indent=2))

    if backend_selection_by_backend:
        print("\nBackend selection summary:")
        for backend, count in backend_selection_by_backend.most_common(top_n):
            print(f"  backend={backend}: count={count}")

        print("\nBackend selection reasons:")
        for reason, count in backend_selection_by_reason.most_common(top_n):
            print(f"  reason={reason}: count={count}")

        print("\nBackend selection by op/backend:")
        for (op_name, backend), count in backend_selection_by_op_backend.most_common(top_n):
            print(f"  op={op_name} backend={backend}: count={count}")

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
            if kernel_filter and filter_lower not in kernel.lower():
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
            if kernel_filter and filter_lower not in op_name.lower():
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
            if kernel_filter and filter_lower not in base_name.lower():
                continue
            base_totals.append((base_name, summarize(samples)))
        if not base_totals:
            print("  (no kernels matched)")
        else:
            base_totals.sort(key=lambda item: item[1]["total_ms"], reverse=True)
            for base_name, stats in base_totals[:top_n]:
                print(render_summary(base_name, stats, indent=2))

    if foundry_capture_kernel_counts:
        print("\nFoundry capture kernel counts:")
        total_dispatches = sum(foundry_capture_dispatches)
        if total_dispatches:
            print(f"  total dispatches (reported): {total_dispatches}")
        for name, count in foundry_capture_kernel_counts.most_common(top_n):
            print(f"  {name}: {count}")

    if matmul_backend_samples:
        print("\nMatmul backend summary:")
        backend_totals = []
        for backend, samples in matmul_backend_samples.items():
            if kernel_filter and filter_lower not in backend.lower():
                continue
            backend_totals.append((backend, summarize(samples)))
        backend_totals.sort(key=lambda item: item[1]["total_ms"], reverse=True)
        for backend, stats in backend_totals[:top_n]:
            title = f"backend={backend}"
            print(render_summary(title, stats, indent=2))

    if matmul_variant_samples:
        print("\nMatmul op/backend summary:")
        variant_totals = []
        for (op_variant, backend), samples in matmul_variant_samples.items():
            label = f"{op_variant} @ {backend}"
            if kernel_filter and filter_lower not in label.lower():
                continue
            variant_totals.append((label, summarize(samples)))
        variant_totals.sort(key=lambda item: item[1]["total_ms"], reverse=True)
        for label, stats in variant_totals[:top_n]:
            print(render_summary(label, stats, indent=2))

    if matmul_shape_samples:
        print("\nMatmul shape backend summary:")
        shape_entries: List[Tuple[str, float, Dict[str, Dict[str, float]]]] = []
        for shape, backend_map in matmul_shape_samples.items():
            backend_stats = {backend: summarize(samples) for backend, samples in backend_map.items()}
            total_ms = sum(stats["total_ms"] for stats in backend_stats.values())
            shape_entries.append((shape, total_ms, backend_stats))
        shape_entries.sort(key=lambda item: item[1], reverse=True)
        printed = 0
        for shape, _total_ms, backend_stats in shape_entries:
            if kernel_filter and filter_lower not in shape.lower() and not any(
                filter_lower in backend.lower() for backend in backend_stats.keys()
            ):
                continue
            if printed >= top_n:
                break
            printed += 1
            print(f"  {shape}:")
            backend_totals = sorted(
                backend_stats.items(), key=lambda item: item[1]["total_ms"], reverse=True
            )
            for backend, stats in backend_totals:
                print(render_summary(f"backend={backend}", stats, indent=4))

    if gpu_wait_samples:
        print("\nGPU wait breakdown:")
        gpu_totals = [
            (op_name, summarize(samples)) for op_name, samples in gpu_wait_samples.items()
        ]
        gpu_totals.sort(key=lambda item: item[1]["total_ms"], reverse=True)
        for op_name, stats in gpu_totals[:top_n]:
            print(render_summary(op_name, stats, indent=2))

    if cb_gpu_samples:
        print("\nCommand buffer GPU timing:")
        cb_totals = [
            (op_name, summarize(samples)) for op_name, samples in cb_gpu_samples.items()
        ]
        cb_totals.sort(key=lambda item: item[1]["total_ms"], reverse=True)
        for op_name, stats in cb_totals[:top_n]:
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
