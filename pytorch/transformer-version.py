from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF"
filename = "qwen2.5-coder-0.5b-instruct-fp16.gguf"

tokenizer = AutoTokenizer.from_pretrained("/Volumes/2TB/test-burn/models", gguf_file=filename)
model = AutoModelForCausalLM.from_pretrained("/Volumes/2TB/test-burn/models", gguf_file=filename)

messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a Python function to compute the factorial of a number."}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=200,
)

# Decode and print the generated text, skipping special tokens
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
