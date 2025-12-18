rm -rf /tmp/profiling*
echo "Profiling enabled FP16"
METALLIC_METRICS_JSONL_PATH=/tmp/profilingenabled-fp16.jsonl METALLIC_ENABLE_PROFILING=true cargo run -q --message-format=short --release -- ./models/qwen2.5-coder-0.5b-instruct-fp16.gguf "create a short js fibonacci function" --output-format=text --max-tokens=100

echo "Profiling enabled Q8"
METALLIC_METRICS_JSONL_PATH=/tmp/profilingenabled-q8.jsonl METALLIC_ENABLE_PROFILING=true cargo run -q --message-format=short --release -- ./models/qwen2.5-coder-0.5b-instruct-q8_0.gguf "create a short js fibonacci function" --output-format=text --max-tokens=100

echo "Analyze profiling enabled FP16"
python tools/analyze_jsonl.py /tmp/profilingenabled-q8.jsonl > q8-profilingenabled.txt

echo "Analyze profiling enabled FP16"
python tools/analyze_jsonl.py /tmp/profilingenabled-fp16.jsonl > fp16-profilingenabled.txt
