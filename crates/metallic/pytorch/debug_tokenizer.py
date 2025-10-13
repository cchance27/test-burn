from transformers import AutoTokenizer, AutoModelForCausalLM
import os

models_dir = "/Volumes/2TB/test-burn/models"
gguf_file = "qwen2.5-coder-0.5b-instruct-fp16.gguf"

tokenizer = AutoTokenizer.from_pretrained(models_dir, gguf_file=gguf_file)
model = AutoModelForCausalLM.from_pretrained(models_dir, gguf_file=gguf_file)

dump_path = "/Volumes/2TB/test-burn/pytorch/dumps/qwen25_20250924_191143"
with open(os.path.join(dump_path, "input_text.txt"), "r") as f:
    text = f.read().strip()

tokens = tokenizer.encode(text)
print("PyTorch tokens:", tokens)

# Also print the first token string
if tokens:
    print("First token string:", tokenizer.decode([tokens[0]]))

# Print vocab size
print("Vocab size:", tokenizer.vocab_size)

# Print embedding weight shape and sample for first token
print("Embedding weight shape:", model.model.embed_tokens.weight.shape)
if tokens:
    first_token = tokens[0]
    print("Weight for token {} first 10:".format(first_token), model.model.embed_tokens.weight[first_token, :10].tolist())
    print("Weight for token 0 first 10:", model.model.embed_tokens.weight[0, :10].tolist())
