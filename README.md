# Metallic Framework

I'm in the process of writing a HIGHLY experimental Apple Metal framework so that i can play with implementing model inference, and experimental optimization.

Sadly we're at the stage of "get it working" still, we're looking to first try to match llamacpp and pytorch performance. 

Using only actively maintained crates, non-framework (no burn, no candle, etc), custom GGUF reading via mmap.

Only supporting Apple Metal hence the name... Metallic.

Right now we're testing with Qwen2.5 (0.5b and 3b)

Qwen2.5b is testing at ~56tok/s vs lmstudio's 150tok/s

So we've got a long ways to go.

