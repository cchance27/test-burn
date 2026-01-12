// crates/metallic/build.rs
use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    // --- Detect build mode and feature flags ---
    let src_kernels = env::var("CARGO_FEATURE_SRC_KERNELS").is_ok();
    let built_kernels = env::var("CARGO_FEATURE_BUILT_KERNELS").is_ok();
    let is_release = env::var("PROFILE").unwrap_or_default() == "release"; // unused now but good context

    // --- Determine which mode we’re in ---
    let use_metallib = built_kernels || (!src_kernels && is_release);

    println!("cargo:rerun-if-env-changed=PROFILE");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_SRC_KERNELS");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_BUILT_KERNELS");

    // --- Always rerun if kernel sources change ---
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // Watch relevant directories
    let kernel_root = crate_dir.join("src/kernels");
    println!("cargo:rerun-if-changed={}", kernel_root.display());

    let metals_root = crate_dir.join("src/metals");
    if metals_root.exists() {
        println!("cargo:rerun-if-changed={}", metals_root.display());
    }

    // --- Only compile .metallib if we’re in metallib mode ---
    if use_metallib {
        println!("Building precompiled Metal libraries (.metallib)");
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let module_cache = out_dir.join("metal_module_cache");
        let _ = fs::create_dir_all(&module_cache);

        // --- 1. Legacy Kernels (kernel.sources manifest logic) ---
        // (Keeping exact logic as requested for parity)
        {
            #[derive(Default)]
            struct KernelInfo {
                metal_path: Option<PathBuf>,
                sources_manifest: Option<PathBuf>,
                rel_dir: Option<PathBuf>,
            }
            let mut kernel_map = std::collections::BTreeMap::<PathBuf, KernelInfo>::new();
            for entry in walkdir::WalkDir::new(&kernel_root)
                .into_iter()
                .filter_map(Result::ok)
                .filter(|e| e.file_type().is_file() && (e.file_name() == "kernel.metal" || e.file_name() == "kernel.sources"))
            {
                let file_name = entry.file_name();
                let parent = entry.path().parent().unwrap().to_path_buf();
                let info = kernel_map.entry(parent.clone()).or_default();
                info.rel_dir = Some(parent.strip_prefix(&kernel_root).unwrap().to_path_buf());
                if file_name == "kernel.metal" {
                    info.metal_path = Some(entry.path().to_path_buf());
                } else {
                    info.sources_manifest = Some(entry.path().to_path_buf());
                }
            }
            for (_dir, info) in kernel_map {
                let rel_dir = info.rel_dir.unwrap();
                let name = rel_dir.to_string_lossy().replace("\\", "/").replace("/", "_"); // normalize

                let metal_input = if let Some(sources_manifest) = info.sources_manifest {
                    let generated = out_dir.join(format!("{name}_includes.metal"));
                    let list = fs::read_to_string(&sources_manifest).expect("kernel.sources readable");
                    let mut includes = String::new();
                    for line in list.lines() {
                        let trimmed = line.trim();
                        if trimmed.is_empty() || trimmed.starts_with('#') {
                            continue;
                        }
                        let part_path = kernel_root.join(trimmed);
                        let abs_path = fs::canonicalize(&part_path).unwrap_or_else(|err| {
                            panic!("Failed to canonicalize '{}': {err}", part_path.display());
                        });
                        includes.push_str(&format!(r#"#include "{}""#, abs_path.display()));
                        includes.push('\n');
                    }
                    fs::write(&generated, includes).expect("write generated include kernel");
                    generated
                } else if let Some(metal) = info.metal_path {
                    metal
                } else {
                    continue;
                };

                compile_metal(&metal_input, &name, &out_dir, &module_cache);
            }
        }

        // --- 2. New Foundry Kernels ---
        // NOTE: Foundry kernels in `metals/` are NOT precompiled because they rely on
        // struct definitions injected at runtime via struct_defs().
        // This keeps the Metal files clean (no duplicate struct defs) and avoids divergence.
        // Foundry kernels are compiled at runtime when first dispatched.
        //
        // If precompilation becomes needed in the future, the build.rs would need to:
        // 1. Parse the Rust structs with #[derive(MetalStruct)]
        // 2. Generate the Metal struct definitions
        // 3. Prepend them to the Metal source before compilation
    } else {
        println!("Skipping metallib build (using source kernels)");
    }
}

fn compile_metal(input_path: &std::path::Path, output_name: &str, out_dir: &std::path::Path, module_cache: &std::path::Path) {
    let air = out_dir.join(format!("{output_name}.air"));
    let metallib = out_dir.join(format!("{output_name}.metallib"));

    if let Some(parent) = air.parent() {
        fs::create_dir_all(parent).expect("create directories for .air output");
    }
    if let Some(parent) = metallib.parent() {
        fs::create_dir_all(parent).expect("create directories for .metallib output");
    }

    // .metal → .air
    let mut metal_cmd = Command::new("xcrun");
    let cache_flag = format!("-fmodules-cache-path={}", module_cache.display());
    metal_cmd.args([
        "-sdk",
        "macosx",
        "metal",
        "-fblocks",
        cache_flag.as_str(),
        "-c",
        input_path.to_str().unwrap(),
        "-o",
        air.to_str().unwrap(),
    ]);
    metal_cmd.env("CLANG_MODULE_CACHE_PATH", module_cache);
    let status = metal_cmd.status().expect("run metal compiler");
    if !status.success() {
        panic!("Command failed: xcrun metal {} -> {}", input_path.display(), air.display());
    }
    // .air → .metallib
    let status = Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metallib",
            air.to_str().unwrap(),
            "-o",
            metallib.to_str().unwrap(),
        ])
        .status()
        .expect("run metallib linker");

    if !status.success() {
        panic!("Command failed: metallib link {}", metallib.display());
    }
}
