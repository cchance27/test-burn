//! Benchmark for the BPE tokenizer implementation

use criterion::{Criterion, criterion_group, criterion_main};
use test_burn::gguf::GGUFFile;
use test_burn::metallic::Tokenizer;

fn load_tokenizer() -> Tokenizer {
    let path = "/Volumes/2TB/test-burn/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf";
    let gguf = GGUFFile::load(path).expect("Failed to load GGUF file");
    Tokenizer::from_gguf_metadata(&gguf.metadata).expect("Failed to create tokenizer")
}

fn benchmark_short_encoding(c: &mut Criterion) {
    let tokenizer = load_tokenizer();
    let short_text = "Hello world!";

    let mut group = c.benchmark_group("short_encoding");
    group.bench_function("serial", |b| {
        b.iter(|| {
            tokenizer
                .encode_serial(short_text)
                .expect("Serial encoding failed");
        })
    });
    group.bench_function("simd", |b| {
        b.iter(|| {
            tokenizer
                .encode_simd(short_text)
                .expect("SIMD encoding failed");
        })
    });
    group.bench_function("parallel", |b| {
        b.iter(|| {
            tokenizer
                .encode_parallel(short_text)
                .expect("Parallel encoding failed");
        })
    });
    group.bench_function("simd_parallel", |b| {
        b.iter(|| {
            tokenizer
                .encode_simd_parallel(short_text)
                .expect("SIMD parallel encoding failed");
        })
    });
    group.finish();
}

fn benchmark_medium_encoding(c: &mut Criterion) {
    let tokenizer = load_tokenizer();
    let medium_text = "This is a medium length text for benchmarking purposes. It contains multiple words and punctuation.";

    let mut group = c.benchmark_group("medium_encoding");
    group.bench_function("serial", |b| {
        b.iter(|| {
            tokenizer
                .encode_serial(medium_text)
                .expect("Serial encoding failed");
        })
    });
    group.bench_function("simd", |b| {
        b.iter(|| {
            tokenizer
                .encode_simd(medium_text)
                .expect("SIMD encoding failed");
        })
    });
    group.bench_function("parallel", |b| {
        b.iter(|| {
            tokenizer
                .encode_parallel(medium_text)
                .expect("Parallel encoding failed");
        })
    });
    group.bench_function("simd_parallel", |b| {
        b.iter(|| {
            tokenizer
                .encode_simd_parallel(medium_text)
                .expect("SIMD parallel encoding failed");
        })
    });
    group.finish();
}

fn benchmark_long_encoding(c: &mut Criterion) {
    let tokenizer = load_tokenizer();
    let long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. ".repeat(10);

    let mut group = c.benchmark_group("long_encoding");
    group.bench_function("serial", |b| {
        b.iter(|| {
            tokenizer
                .encode_serial(&long_text)
                .expect("Serial encoding failed");
        })
    });
    group.bench_function("simd", |b| {
        b.iter(|| {
            tokenizer
                .encode_simd(&long_text)
                .expect("SIMD encoding failed");
        })
    });
    group.bench_function("parallel", |b| {
        b.iter(|| {
            tokenizer
                .encode_parallel(&long_text)
                .expect("Parallel encoding failed");
        })
    });
    group.bench_function("simd_parallel", |b| {
        b.iter(|| {
            tokenizer
                .encode_simd_parallel(&long_text)
                .expect("SIMD parallel encoding failed");
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    benchmark_short_encoding,
    benchmark_medium_encoding,
    benchmark_long_encoding,
);
criterion_main!(benches);
