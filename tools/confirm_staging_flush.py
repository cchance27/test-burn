#!/usr/bin/env python3
import re
from pathlib import Path

def extract_stats(path: Path):
    total = None
    avg = None
    count = None
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if "tensor_staging_flush:" in line:
            # Example: "  tensor_staging_flush: count=536 | total=179.28ms | avg=0.334ms | ..."
            m_total = re.search(r"total=([0-9.]+)ms", line)
            m_avg = re.search(r"avg=([0-9.]+)ms", line)
            m_cnt = re.search(r"count=(\d+)", line)
            if m_total: total = float(m_total.group(1))
            if m_avg: avg = float(m_avg.group(1))
            if m_cnt: count = int(m_cnt.group(1))
            break
    return {"count": count, "total_ms": total, "avg_ms": avg}

def main():
    disabled = Path("profiling_disabled_jsonl_analyzed.txt")
    enabled = Path("profiling_enabled_jsonl_analyzed.txt")
    d = extract_stats(disabled)
    e = extract_stats(enabled)
    print("tensor_staging_flush summary:")
    if d:
        print(f"  disabled: count={d['count']} total={d['total_ms']}ms avg={d['avg_ms']}ms")
    else:
        print("  disabled: not found")
    if e:
        print(f"  enabled:  count={e['count']} total={e['total_ms']}ms avg={e['avg_ms']}ms")
    else:
        print("  enabled:  not found")

if __name__ == "__main__":
    main()

