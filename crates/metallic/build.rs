// crates/metallic/build.rs
use std::{env, path::PathBuf, process::Command};

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

        for entry in walkdir::WalkDir::new(&kernel_root)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_name() == "kernel.metal")
        {
            let src = entry.path();
            let name = src.parent().unwrap().file_name().unwrap().to_str().unwrap();
            let air = out_dir.join(format!("{name}.air"));
            let metallib = out_dir.join(format!("{name}.metallib"));

            // .metal → .air
            run(
                "xcrun",
                &["-sdk", "macosx", "metal", "-c", src.to_str().unwrap(), "-o", air.to_str().unwrap()],
            );
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
