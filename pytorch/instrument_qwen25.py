import argparse
import os
import json
from datetime import datetime
from typing import Dict, Any

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def tensor_stats(t: torch.Tensor, sample_elems: int = 8) -> Dict[str, Any]:
    t_cpu = t.detach().float().cpu()
    flat = t_cpu.flatten()
    stats = {
        "shape": list(t_cpu.shape),
        "dtype": str(t_cpu.dtype),
        "min": float(flat.min().item()) if flat.numel() > 0 else None,
        "max": float(flat.max().item()) if flat.numel() > 0 else None,
        "mean": float(flat.mean().item()) if flat.numel() > 0 else None,
        "std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
        "sample": flat[:sample_elems].tolist(),
    }
    return stats


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Instrument Qwen2.5 GGUF (PyTorch/Transformers) to dump per-layer tensors for comparison with Rust implementation.")
    parser.add_argument("--models-dir", type=str, default="/Volumes/2TB/test-burn/models", help="Directory containing the GGUF model")
    parser.add_argument("--gguf-file", type=str, default="qwen2.5-coder-0.5b-instruct-fp16.gguf", help="GGUF filename inside models-dir")
    parser.add_argument("--system", type=str, default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", help="System message content")
    parser.add_argument("--user", type=str, default="Write a python script that prints hello world", help="User prompt content")
    parser.add_argument("--max-new-tokens", type=int, default=0, help="If >0, run generate() for this many tokens; otherwise run a single forward pass only")
    parser.add_argument("--layer-limit", type=int, default=-1, help="If >0, only instrument the first N layers")
    parser.add_argument("--save-arrays", action="store_true", help="Also save full tensors as .npy files in addition to JSON stats")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory; default is pytorch/dumps/<timestamp>")
    parser.add_argument("--print-output", action="store_true", help="Print decoded model output text to stdout and save to file")
    args = parser.parse_args()

    models_dir = args.models_dir
    gguf_file = args.gguf_file

    tokenizer = AutoTokenizer.from_pretrained(models_dir, gguf_file=gguf_file)
    model = AutoModelForCausalLM.from_pretrained(models_dir, gguf_file=gguf_file)
    model.eval()

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.user},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or os.path.join(os.path.dirname(__file__), "dumps", f"qwen25_{timestamp}")
    ensure_dir(outdir)

    # Collector structure
    collected: Dict[str, torch.Tensor] = {}

    def record(name: str, tensor: torch.Tensor):
        # respect layer-limit for per-layer captures
        if name.startswith("layer_") and args.layer_limit > 0:
            try:
                # name like layer_3/... -> get 3
                layer_idx = int(name.split("_", 1)[1].split("/", 1)[0])
                if layer_idx >= args.layer_limit:
                    return
            except Exception:
                pass
        # Only keep first occurrence for deterministic prompt-only forward
        if name not in collected:
            collected[name] = tensor.detach()

    # Try to locate common submodules robustly
    base = getattr(model, "model", None) or getattr(model, "transformer", None) or model

    # Embedding
    embed = getattr(base, "embed_tokens", None)
    if embed is not None:
        embed.register_forward_hook(lambda m, inp, out: record("embeddings", out))

    # Final norm (varies by architecture)
    final_norm = getattr(base, "norm", None) or getattr(base, "final_layernorm", None) or getattr(base, "ln_f", None)
    if final_norm is not None:
        final_norm.register_forward_hook(lambda m, inp, out: record("final_norm", out))

    # LM head
    lm_head = getattr(model, "lm_head", None) or getattr(model, "embed_out", None)
    if lm_head is not None:
        lm_head.register_forward_hook(lambda m, inp, out: record("logits_pre_softmax", out))

    # Layers
    layers = getattr(base, "layers", None) or getattr(base, "h", None) or getattr(base, "decoder_layers", None)
    if layers is None:
        raise RuntimeError("Could not find decoder layers on model; inspect model structure.")

    # Try to find typical submodule names inside each layer
    for i, layer in enumerate(layers):
        # Norms
        in_norm = getattr(layer, "input_layernorm", None) or getattr(layer, "rms_1", None) or getattr(layer, "ln_1", None)
        if in_norm is not None:
            in_norm.register_forward_hook(lambda m, inp, out, idx=i: record(f"layer_{idx}/attn_norm", out))

        post_norm = getattr(layer, "post_attention_layernorm", None) or getattr(layer, "rms_2", None) or getattr(layer, "ln_2", None)
        if post_norm is not None:
            post_norm.register_forward_hook(lambda m, inp, out, idx=i: record(f"layer_{idx}/mlp_norm", out))

        # Attention block and its projections
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None) or getattr(layer, "attn", None)
        if attn is not None:
            q_proj = getattr(attn, "q_proj", None)
            k_proj = getattr(attn, "k_proj", None)
            v_proj = getattr(attn, "v_proj", None)
            o_proj = getattr(attn, "o_proj", None) or getattr(attn, "out_proj", None)
            if q_proj is not None:
                q_proj.register_forward_hook(lambda m, inp, out, idx=i: record(f"layer_{idx}/q_proj_out", out))
            if k_proj is not None:
                k_proj.register_forward_hook(lambda m, inp, out, idx=i: record(f"layer_{idx}/k_proj_out", out))
            if v_proj is not None:
                v_proj.register_forward_hook(lambda m, inp, out, idx=i: record(f"layer_{idx}/v_proj_out", out))
            if o_proj is not None:
                o_proj.register_forward_hook(lambda m, inp, out, idx=i: record(f"layer_{idx}/attn_out", out))

        # MLP
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            mlp.register_forward_hook(lambda m, inp, out, idx=i: record(f"layer_{idx}/mlp_out", out))
            gate_proj = getattr(mlp, "gate_proj", None)
            up_proj = getattr(mlp, "up_proj", None)
            down_proj = getattr(mlp, "down_proj", None)
            if hasattr(mlp, 'act_fn') and mlp.act_fn is not None:
                # For SwiGLU, override forward to capture intermediates
                mlp.layer_idx = i
                def custom_forward(self, x):
                    gate = self.gate_proj(x)
                    up = self.up_proj(x)
                    record(f"layer_{self.layer_idx}/gate_proj_out", gate)
                    record(f"layer_{self.layer_idx}/up_proj_out", up)
                    silu_gate = torch.nn.functional.silu(gate)
                    mul_out = silu_gate * up
                    down_out = self.down_proj(mul_out)
                    record(f"layer_{self.layer_idx}/silu_out", silu_gate)
                    record(f"layer_{self.layer_idx}/mul_out", mul_out)
                    return down_out
                mlp.forward = custom_forward.__get__(mlp, type(mlp))
                if down_proj is not None:
                    down_proj.register_forward_hook(lambda m, inp, out, idx=i: record(f"layer_{idx}/down_proj_out", out))

    generated_text: str | None = None

    with torch.no_grad():
        if args.max_new_tokens and args.max_new_tokens > 0:
            # Also set config to expose useful internals if available
            if hasattr(model.config, "output_hidden_states"):
                model.config.output_hidden_states = True
            if hasattr(model.config, "output_attentions"):
                model.config.output_attentions = True

            gen_out = model.generate(
                inputs.input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
            # The first-step logits are also available from scores[0]
            if hasattr(gen_out, "scores") and len(gen_out.scores) > 0:
                record("logits_step0", gen_out.scores[0])
            # Decode and optionally print/save
            gen_ids = getattr(gen_out, "sequences", None)
            if gen_ids is None:
                gen_ids = gen_out
            generated_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
        else:
            # Single forward pass (no generation), better for one-to-one comparison
            fw = model(**inputs, use_cache=False, output_hidden_states=True, output_attentions=True, return_dict=True)
            record("logits", fw.logits)
            if getattr(fw, "hidden_states", None) is not None:
                # Last hidden state prior to lm_head
                record("hidden_states_last", fw.hidden_states[-1])
            # If requested, run a short generation only for printing/validation
            if args.print_output:
                gen_out = model.generate(
                    inputs.input_ids,
                    max_new_tokens=128,
                    do_sample=False,
                )
                generated_text = tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0]

    # Save outputs
    stats = {name: tensor_stats(t) for name, t in collected.items()}
    with open(os.path.join(outdir, "stats.json"), "w") as f:
        json.dump({
            "meta": {
                "gguf_file": gguf_file,
                "prompt": args.user,
                "system": args.system,
                "mode": "generate" if (args.max_new_tokens and args.max_new_tokens > 0) else "forward_only",
            },
            "stats": stats,
        }, f, indent=2)

    if args.save_arrays:
        arrays_dir = os.path.join(outdir, "arrays")
        ensure_dir(arrays_dir)
        for name, t in collected.items():
            # replace path separators
            safe_name = name.replace("/", "__")
            np.save(os.path.join(arrays_dir, f"{safe_name}.npy"), t.detach().float().cpu().numpy())

    # Also dump the tokenized text for reproducibility
    with open(os.path.join(outdir, "input_text.txt"), "w") as f:
        f.write(text)

    if generated_text is not None:
        with open(os.path.join(outdir, "generated_text.txt"), "w") as f:
            f.write(generated_text)
        # Print a concise confirmation to stdout
        print("===== Model Generated Output (truncated) =====")
        preview = generated_text[:800]
        print(preview)
        if len(generated_text) > len(preview):
            print("... [truncated]")

    print(f"Instrumentation complete. Wrote outputs to: {outdir}")


if __name__ == "__main__":
    main()