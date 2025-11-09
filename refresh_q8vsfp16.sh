rm -rf /tmp/profiling*
METALLIC_METRICS_JSONL_PATH=/tmp/profilingenabled-fp16.jsonl METALLIC_ENABLE_PROFILING=true cargo run -q --message-format=short --release -- ./models/qwen2.5-coder-0.5b-instruct-fp16.gguf "create a short js fibonacci function" --output-format=text --max-tokens=100
METALLIC_METRICS_JSONL_PATH=/tmp/profilingenabled-q8.jsonl METALLIC_ENABLE_PROFILING=true cargo run -q --message-format=short --release -- ./models/qwen2.5-coder-0.5b-instruct-q8_0.gguf "create a short js fibonacci function" --output-format=text --max-tokens=100
python analyze_jsonl.py /tmp/profilingenabled-q8.jsonl > q8-profilingenabled.txt
python analyze_jsonl.py /tmp/profilingenabled-fp16.jsonl > fp16-profilingenabled.txt
