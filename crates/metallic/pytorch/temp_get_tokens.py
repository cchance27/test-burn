from transformers import AutoTokenizer
import os
models_dir = "/Volumes/2TB/test-burn/models"
gguf_file = "qwen2.5-coder-0.5b-instruct-fp16.gguf"
tokenizer = AutoTokenizer.from_pretrained(models_dir, gguf_file=gguf_file)
dump_path = "dumps/qwen25_20250924_191143"
with open(os.path.join(dump_path, "input_text.txt"), "r") as f:
    text = f.read().strip()
tokens = tokenizer.encode(text)
print(tokens)
