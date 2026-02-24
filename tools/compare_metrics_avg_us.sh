#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <current.jsonl> <baseline.jsonl> [top_n]" >&2
  exit 2
fi

CUR="$1"
BASE="$2"
TOP_N="${3:-20}"

if [[ ! -f "$CUR" ]]; then
  echo "error: missing current metrics file: $CUR" >&2
  exit 2
fi
if [[ ! -f "$BASE" ]]; then
  echo "error: missing baseline metrics file: $BASE" >&2
  exit 2
fi

python3 - "$CUR" "$BASE" "$TOP_N" <<'PY'
import json
import sys
from collections import defaultdict

cur_path, base_path, top_n = sys.argv[1], sys.argv[2], int(sys.argv[3])

def agg(path):
    total = defaultdict(int)
    count = defaultdict(int)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            ev = row.get('event', {})
            if ev.get('type') != 'GpuOpCompleted':
                continue
            data = ev.get('data', {})
            op = data.get('op_name')
            if not op or op.startswith('foundry_capture#'):
                continue
            dur = data.get('duration_us')
            if dur is None:
                continue
            try:
                dur = int(dur)
            except Exception:
                continue
            total[op] += dur
            count[op] += 1
    return total, count

cur_total, cur_count = agg(cur_path)
base_total, base_count = agg(base_path)
ops = sorted(set(cur_total) | set(base_total))

rows = []
for op in ops:
    ct = cur_total.get(op, 0)
    cn = cur_count.get(op, 0)
    bt = base_total.get(op, 0)
    bn = base_count.get(op, 0)
    cavg = (ct / cn) if cn else 0.0
    bavg = (bt / bn) if bn else 0.0
    rows.append((cavg - bavg, cavg, bavg, cn, bn, op))

rows.sort(key=lambda x: x[0], reverse=True)

print(f"{'delta_avg':<12} {'cur_avg':<12} {'base_avg':<12} {'cur_n':<8} {'base_n':<8} op_name")
for delta, cavg, bavg, cn, bn, op in rows[:top_n]:
    print(f"{delta:<12.2f} {cavg:<12.2f} {bavg:<12.2f} {cn:<8d} {bn:<8d} {op}")
PY
