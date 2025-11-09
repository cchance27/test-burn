// crates/metallic/build.rs
use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    // --- Detect build mode and feature flags ---
    let src_kernels = env::var("CARGO_FEATURE_SRC_KERNELS").is_ok();
    let built_kernels = env::var("CARGO_FEATURE_BUILT_KERNELS").is_ok();
    let is_release = env::var("PROFILE").unwrap_or_default() == "release";

    // --- Determine which mode we’re in ---
    let use_metallib = built_kernels || (!src_kernels && is_release);

    println!("cargo:rerun-if-env-changed=PROFILE");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_SRC_KERNELS");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_BUILT_KERNELS");

    // --- Always rerun if kernel sources change ---
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernel_root = crate_dir.join("src/kernels");
    println!("cargo:rerun-if-changed={}", kernel_root.display());

    // --- Only compile .metallib if we’re in metallib mode ---
    if use_metallib {
        println!("Building precompiled Metal libraries (.metallib)");
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let module_cache = out_dir.join("metal_module_cache");
        let _ = fs::create_dir_all(&module_cache);

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
            let name = rel_dir.to_string_lossy().replace("\\", "/");
            let air = out_dir.join(format!("{name}.air"));
            let metallib = out_dir.join(format!("{name}.metallib"));

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
                panic!("Kernel directory '{}' has no kernel.metal or kernel.sources", name);
            };

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
                metal_input.to_str().unwrap(),
                "-o",
                air.to_str().unwrap(),
            ]);
            metal_cmd.env("CLANG_MODULE_CACHE_PATH", &module_cache);
            let status = metal_cmd.status().unwrap();
            if !status.success() {
                panic!("Command failed: xcrun metal {} -> {}", metal_input.display(), air.display());
            }
            // .air → .metallib
            run(
                "xcrun",
                &[
                    "-sdk",
                    "macosx",
                    "metallib",
                    air.to_str().unwrap(),
                    "-o",
                    metallib.to_str().unwrap(),
                ],
            );
        }
    } else {
        println!("Skipping metallib build (using source kernels)");
    }
}

fn run(cmd: &str, args: &[&str]) {
    let status = Command::new(cmd).args(args).status().unwrap();
    if !status.success() {
        panic!("Command failed: {} {:?}", cmd, args);
    }
}
