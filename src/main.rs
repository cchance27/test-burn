use std::env;
use std::process;

use test_burn::metallic::generation::GenerationConfig;

fn main() {
    // Minimal CLI:
    //   cargo run -- /path/to/model.gguf [PROMPT] [--diag]
    //   or: cargo run -- /path/to/model.gguf --diag
    let mut args = env::args().skip(1);
    let gguf_path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("Usage: cargo run -- <GGUF_PATH> [PROMPT]");
            process::exit(1);
        }
    };

    let prompt = args.next().unwrap_or_else(|| "Hello World".to_string());

    println!("Loading GGUF file: {}", gguf_path);
    // Load GGUF (memory-mapped)
    let gguf = match test_burn::gguf::GGUFFile::load(&gguf_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to load GGUF file '{}': {:?}", gguf_path, e);
            process::exit(2);
        }
    };

    // Initialize Metallic context (Metal / MPS)
    let mut ctx = match test_burn::metallic::Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to create Metallic Context: {:?}", e);
            process::exit(3);
        }
    };

    // Build GGUFModel using the loader
    let loader = test_burn::gguf::model_loader::GGUFModelLoader::new(gguf);
    let gguf_model = match loader.load_model(&ctx) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to create GGUFModel: {:?}", e);
            process::exit(4);
        }
    };

    // Instantiate Qwen25 (or the appropriate LoadableModel) from GGUFModel
    println!("Instantiating Qwen25 from GGUF model (this may allocate device memory)...");
    let mut qwen: test_burn::metallic::qwen25::Qwen25 = match gguf_model.instantiate(&mut ctx) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Failed to instantiate Qwen25 from GGUFModel: {:?}", e);
            process::exit(5);
        }
    };

    // Try to construct a tokenizer from GGUF metadata (best-effort)
    let tokenizer = match test_burn::metallic::tokenizer::Tokenizer::from_gguf_metadata(
        &gguf_model.metadata,
    ) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "Warning: failed to build tokenizer from GGUF metadata: {:?}. Falling back to basic whitespace tokenizer.",
                e
            );
            // Very small fallback tokenizer: map bytes to token ids 0..255
            let mut map = rustc_hash::FxHashMap::default();
            for i in 0..256u32 {
                map.insert(i, format!("<{}>", i));
            }
            test_burn::metallic::tokenizer::Tokenizer::from_vocab_and_merges(map, vec![])
                .expect("fallback tokenizer")
        }
    };

    // Tokenize prompt
    println!("Tokenizing prompt: {:?}", prompt);
    let tokens = match tokenizer.encode(&prompt) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to tokenize prompt: {:?}", e);
            process::exit(6);
        }
    };

    println!("Token count: {}", tokens.len());

    let cfg = GenerationConfig {
        max_tokens: 40,
        temperature: 0.7,
        top_p: 0.95,
        top_k: 40,
    };

    match test_burn::metallic::generation::generate(&mut qwen, &tokenizer, &mut ctx, &prompt, &cfg)
    {
        Ok(out_text) => {
            println!("Generation result:\n{}", out_text);
            process::exit(0);
        }
        Err(e) => {
            eprintln!("Generation failed: {:?}", e);
            process::exit(8);
        }
    }
}
