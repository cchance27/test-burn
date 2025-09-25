#!/usr/bin/env python3
import os
import sys
import json
import ast
import argparse
import numpy as np

def load_debug_txt(path):
    """
    Expect two-line file:
    1 | [dim0, dim1, ...]
    2 | comma,separated,floats,...
    """
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().strip()
        # skip any empty leading lines
        while first == "":
            first = f.readline().strip()
        second = f.readline().strip()
        # If second is empty, try to read the rest (some dumps may wrap lines)
        if second == "":
            rest = f.read().strip()
            second = rest
    try:
        dims = ast.literal_eval(first)
    except Exception:
        # try to extract bracket content
        s = first
        l = s.find("[")
        r = s.rfind("]")
        if l != -1 and r != -1:
            dims = ast.literal_eval(s[l:r+1])
        else:
            raise
    # parse numbers (handle possible trailing commas)
    # Some dumps may have spaces after commas
    parts = second.split(",")
    vals = []
    for p in parts:
        ps = p.strip()
        if ps == "":
            continue
        try:
            vals.append(float(ps))
        except:
            # ignore non-numeric tails
            try:
                # sometimes there are hex or other stray tokens; attempt to clean
                vals.append(float(ps.split()[0]))
            except:
                pass
    arr = np.array(vals, dtype=np.float32)
    expected = 1
    for d in dims:
        expected *= d
    if arr.size != expected:
        raise ValueError(f"Parsed {arr.size} elements but dims {dims} expect {expected} for file {path}")
    arr = arr.reshape(tuple(dims))
    return arr

def try_transforms_and_report(rust, py, name="ffn_output", outdir="."):
    res = {}
    def metrics(a, b):
        diff = a - b
        ad = np.abs(diff)
        return {
            "max_abs": float(ad.max()),
            "mean_abs": float(ad.mean()),
            "std_abs": float(ad.std()),
            "num_gt_1e-3": int((ad > 1e-3).sum()),
            "num_gt_1e-4": int((ad > 1e-4).sum()),
            "total": int(ad.size),
        }
    res["raw"] = metrics(rust, py)
    # try transpose if shapes match on transpose
    if rust.shape == py.T.shape:
        res["transpose"] = metrics(rust, py.T)
    # try per-dim (feature) bias: compute mean over seq axis (axis=0 if seq is axis0)
    if rust.shape == py.shape:
        bias = (py - rust).mean(axis=0)
        rust_bias_corrected = rust + bias  # adding mean(py-rust) to rust moves towards py
        res["with_estimated_output_bias"] = metrics(rust_bias_corrected, py)
        # also report bias magnitude
        res["estimated_output_bias_stats"] = {
            "bias_mean": float(bias.mean()),
            "bias_std": float(bias.std()),
            "bias_max_abs": float(np.abs(bias).max())
        }
    # try per-seq bias (less likely)
    if rust.shape == py.shape:
        bias_s = (py - rust).mean(axis=1)
        res["estimated_seq_bias_stats"] = {
            "bias_seq_mean": float(bias_s.mean()),
            "bias_seq_std": float(bias_s.std()),
            "bias_seq_max_abs": float(np.abs(bias_s).max())
        }
    # top-k diffs
    diff = (rust - py).ravel()
    idx = np.argsort(np.abs(diff))[::-1]
    topk = min(20, diff.size)
    top = []
    for i in range(topk):
        ind = idx[i]
        top.append({
            "index_flat": int(ind),
            "abs_diff": float(np.abs(diff[ind])),
            "rust": float(rust.ravel()[ind]),
            "py": float(py.ravel()[ind]),
            "diff": float(diff[ind]),
        })
    res["top_diffs"] = top
    # write report json
    out_path = os.path.join(outdir, f"compare_{name}.json")
    with open(out_path, "w", encoding="utf-8") as jf:
        json.dump(res, jf, indent=2)
    # also print concise
    print(f"=== Comparison report for {name} ===")
    print(json.dumps({
        "raw": res["raw"],
        **({"transpose": res["transpose"]} if "transpose" in res else {}),
        **({"with_estimated_output_bias": res.get("with_estimated_output_bias")} if "with_estimated_output_bias" in res else {})
    }, indent=2))
    print(f"Wrote detailed report to {out_path}")
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch-dumps", default="/Volumes/2TB/test-burn/pytorch/dumps/qwen25_20250924_191143/arrays", help="Path to PyTorch .npy arrays")
    parser.add_argument("--rust-dumps", default="debug_dumps", help="Path to Rust debug text dumps")
    parser.add_argument("--outdir", default="debug_compare_out", help="Output directory for reports")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # paths
    py_embeddings_p = os.path.join(args.pytorch_dumps, "embeddings.npy")
    py_attn_out_p = os.path.join(args.pytorch_dumps, "layer_0__attn_out.npy")
    py_mlp_out_p = os.path.join(args.pytorch_dumps, "layer_0__mlp_out.npy")

    for p in (py_embeddings_p, py_attn_out_p, py_mlp_out_p):
        if not os.path.exists(p):
            print(f"ERROR: missing pytorch file {p}", file=sys.stderr)
            sys.exit(2)

    print("Loading PyTorch arrays...")
    py_embeddings = np.load(py_embeddings_p).astype(np.float32)
    py_attn_out = np.load(py_attn_out_p).astype(np.float32)
    py_mlp_out = np.load(py_mlp_out_p).astype(np.float32)

    print("Computing expected FFN from PyTorch dumps: py_mlp_out - (embeddings + attn_out)")
    py_sum = None
    try:
        py_sum = py_embeddings + py_attn_out
    except ValueError:
        # attempt broadcasting if shapes differ
        py_sum = py_embeddings + py_attn_out
    py_ffn = py_mlp_out - py_sum

    # load Rust debug dumps (ffn_output plus optionally gate_proj/up_proj/hidden)
    rust_ffn_p = os.path.join(args.rust_dumps, "ffn_output.txt")
    rust_gate_p = os.path.join(args.rust_dumps, "gate_proj.txt")
    rust_up_p = os.path.join(args.rust_dumps, "up_proj.txt")
    rust_hidden_p = os.path.join(args.rust_dumps, "hidden.txt")

    if not os.path.exists(rust_ffn_p):
        print(f"ERROR: missing rust ffn output dump at {rust_ffn_p}", file=sys.stderr)
        sys.exit(2)

    print("Loading Rust ffn_output...")
    rust_ffn = load_debug_txt(rust_ffn_p)

    # quick shape report
    print("Shapes: py_ffn", py_ffn.shape, "rust_ffn", rust_ffn.shape)

    # attempt direct compare and transforms
    res = try_transforms_and_report(rust_ffn, py_ffn, name="ffn_output", outdir=args.outdir)

    # If gate/up/hidden dumps exist, load and save basic diagnostics
    for label, p in (("gate_proj", rust_gate_p), ("up_proj", rust_up_p), ("hidden", rust_hidden_p)):
        if os.path.exists(p):
            try:
                arr = load_debug_txt(p)
                np.save(os.path.join(args.outdir, f"{label}_rust.npy"), arr)
                print(f"Loaded {label} shape {arr.shape} -> saved to {args.outdir}/{label}_rust.npy")
            except Exception as e:
                print(f"Failed to load {p}: {e}", file=sys.stderr)
        else:
            print(f"{label} dump not found at {p}")

    # Save py_ffn and rust_ffn for offline inspection
    np.save(os.path.join(args.outdir, "py_ffn.npy"), py_ffn)
    np.save(os.path.join(args.outdir, "rust_ffn.npy"), rust_ffn)
    print(f"Saved py_ffn.npy and rust_ffn.npy to {args.outdir}")

if __name__ == "__main__":
    main()