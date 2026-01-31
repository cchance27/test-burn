#!/usr/bin/env python3

import argparse
import subprocess
import sys
import re
import time
from typing import List, Optional, Dict

def run_command(cmd: List[str], capture_output=True) -> subprocess.CompletedProcess:
    """Runs a shell command."""
    return subprocess.run(cmd, text=True, capture_output=capture_output)

def get_commits(n: int) -> List[str]:
    """Gets the last n commit hashes."""
    result = run_command(["git", "log", "-n", str(n), "--pretty=format:%H"])
    if result.returncode != 0:
        print(f"Error getting commits: {result.stderr}", flush=True)
        sys.exit(1)
    return result.stdout.strip().split('\n')

def get_current_branch() -> str:
    """Gets the current git branch."""
    result = run_command(["git", "branch", "--show-current"])
    return result.stdout.strip()

def run_benchmark(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    """Runs the benchmark and parses detailed stats, streaming output."""
    cmd = ["./tools/run_throughput.sh"]
    
    # Pass through arguments
    if args.q8: cmd.append("--q8")
    if args.fp16: cmd.append("--fp16")
    if args.max_tokens: cmd.extend(["--max-tokens", str(args.max_tokens)])
    if args.engine: cmd.extend(["--engine", args.engine])
    if args.iterations: cmd.extend(["--iterations", str(args.iterations)])

    print(f"Running benchmark: {' '.join(cmd)}", flush=True)
    
    metrics = {
        "decode": {"min": 0.0, "avg": 0.0, "max": 0.0},
        "prefill": {"min": 0.0, "avg": 0.0, "max": 0.0}
    }

    # Use Popen to stream output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    output_lines = []
    
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line, end='', flush=True) # Stream to stdout
            output_lines.append(line)
            
            # Parse stats line on the fly
            if "[stats]" in line and "(tps)" in line:
                try:
                    category = ""
                    if "Decode" in line:
                        category = "decode"
                    elif "Prefill" in line:
                        category = "prefill"
                    
                    if category:
                        # parts[0] -> "[stats] Decode (tps)         "
                        # parts[1] -> " min:   187.03 "
                        # parts[2] -> " avg:   187.81 "
                        # parts[3] -> " max:   188.95"
                        parts = line.split('|')
                        for part in parts[1:]:
                            if ':' in part:
                                key, val = part.split(':')
                                key = key.strip()
                                val = float(val.strip())
                                if key in metrics[category]:
                                    metrics[category][key] = val
                except Exception:
                    pass

    return_code = process.poll()
    if return_code != 0:
        print(f"Benchmark failed (exit code {return_code})", flush=True)
        return None

    return metrics

def main():
    
    parser = argparse.ArgumentParser(description="Benchmark past git commits to find performance regressions.")
    parser.add_argument("--commits", type=int, default=5, help="Number of past commits to benchmark")
    parser.add_argument("--q8", action="store_true", help="Run Q8 benchmark")
    parser.add_argument("--fp16", action="store_true", help="Run FP16 benchmark")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens (default: 256)")
    parser.add_argument("--engine", type=str, default="foundry", help="Engine to use (default: foundry)")
    parser.add_argument("--iterations", type=int, default=5, help="Iterations per benchmark (default: 5)")
    parser.add_argument("--reverse", action="store_true", help="Run benchmarks from oldest to newest")
    parser.add_argument("--sleep", type=int, default=0, help="Sleep N seconds between benchmarks to let GPU cool down")
    
    
    args = parser.parse_args()
    

    original_branch = get_current_branch()
    if not original_branch:
        # Detached HEAD, save the commit hash
        result = run_command(["git", "rev-parse", "HEAD"])
        original_branch = result.stdout.strip()
        print(f"Starting in detached HEAD at {original_branch}", flush=True)
    else:
        print(f"Starting on branch {original_branch}", flush=True)

    commits = get_commits(args.commits)
    if args.reverse:
        commits.reverse()
        print("Running in REVERSE order (Oldest -> Newest)", flush=True)

    results = []

    print(f"Benchmarking {len(commits)} commits...", flush=True)
    
    try:
        for i, commit in enumerate(commits):
            print(f"\n[{i+1}/{len(commits)}] Checking out {commit[:7]}...", flush=True)
            run_command(["git", "checkout", commit], capture_output=True) 
            
            # Get commit message for display
            msg_res = run_command(["git", "log", "-1", "--pretty=%s", commit])
            msg = msg_res.stdout.strip()
            print(f"Commit: {msg}", flush=True)

            if args.sleep > 0 and i > 0:
                 print(f"Sleeping {args.sleep}s to cool down...", flush=True)
                 time.sleep(args.sleep)

            metrics = run_benchmark(args)
            if metrics:
                s = f"Dec: {metrics['decode']['avg']:.1f} (±{metrics['decode']['max']-metrics['decode']['min']:.1f}) | Pre: {metrics['prefill']['avg']:.1f}"
                print(f"Result: {s}", flush=True)
            else:
                print("Result: Failed", flush=True)
                
            results.append({"commit": commit, "msg": msg, "metrics": metrics})

    except KeyboardInterrupt:
        print("\nInterrupted! Restoring git state...", flush=True)
    finally:
        print(f"\nRestoring {original_branch}...", flush=True)
        run_command(["git", "checkout", original_branch], capture_output=True)

    print("\n" + "="*140, flush=True)
    # Header
    print(f"{'Commit':<9} | {'Decode (TPS)':^26} | {'Prefill (TPS)':^26} | {'Message'}")
    print(f"{'':<9} | {'Min':>8} {'Avg':>8} {'Max':>8} | {'Min':>8} {'Avg':>8} {'Max':>8} |")
    print("-" * 140, flush=True)
    
    for res in results:
        commit_short = res['commit'][:8]
        msg = res['msg'][:60]
        m = res['metrics']
        
        if m:
            d = m['decode']
            p = m['prefill']
            print(f"{commit_short:<9} | {d['min']:8.2f} {d['avg']:8.2f} {d['max']:8.2f} | {p['min']:8.2f} {p['avg']:8.2f} {p['max']:8.2f} | {msg}", flush=True)
        else:
            print(f"{commit_short:<9} | {'FAILED':^26} | {'FAILED':^26} | {msg}", flush=True)
            
    print("="*140, flush=True)

if __name__ == "__main__":
    
    main()
