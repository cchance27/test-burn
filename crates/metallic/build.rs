// crates/metallic/build.rs
use std::{env, fs, path::PathBuf, process::Command};

use syn::{Expr, ItemImpl, Lit, Meta, parse_file, visit::Visit};

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

        // --- 2. New Foundry Kernels (Rust Scanning Logic) ---
        if metals_root.exists() {
            // Walker struct to find #[derive(Kernel)] and extract #[kernel(...)]
            struct KernelVisitor {
                kernels: Vec<FoundryKernel>,
            }
            struct FoundryKernel {
                name: String,
                source: String,
                includes: Vec<String>,
            }
            impl<'ast> Visit<'ast> for KernelVisitor {
                fn visit_item_impl(&mut self, node: &'ast ItemImpl) {
                    // We look for internal macro attributes or just parse the tokens.
                    // But wait, macros consume attributes. If we use `#[derive(Kernel)]`,
                    // the `#[kernel(...)]` attribute *might* be consumed if the macro doesn't re-emit it?
                    // Actually, derive macros don't consume attributes visible to other macros?
                    // Wait, derive macros keep the attributes adjacent usually.
                    // But here we are parsing the SOURCE code, so the attributes are definitely there.

                    // We want to find: impl ... for StructName ...
                    // And check if StructName has #[derive(Kernel)].
                    // OR simpler: we iterate over Structs.
                    // The user's pseudo-code scanned items.
                    // Let's scan for `ItemStruct` instead?
                    // The user pseudo-code used `syn::Item::Fn`? That's for function kernels.
                    // Our `#[derive(Kernel)]` is on a STRUCT.
                    // Let's implement `visit_derive_input` or just traverse items.

                    // Actually strict Syn logic:
                    // iterate items -> check if ItemStruct -> check attrs for derive(Kernel)
                    // -> check attrs for kernel(...)

                    syn::visit::visit_item_impl(self, node);
                }

                fn visit_item_struct(&mut self, node: &'ast syn::ItemStruct) {
                    let has_derive_kernel = node.attrs.iter().any(|attr| {
                        if attr.path().is_ident("derive") {
                            // Parsing derive options is complex, assume simplified check for now or parse properly
                            // This is a comprehensive parser script, let's keep it robust.
                            let mut is_k = false;
                            let _ = attr.parse_nested_meta(|meta| {
                                if meta.path.is_ident("Kernel") {
                                    is_k = true;
                                }
                                Ok(())
                            });
                            is_k
                        } else {
                            false
                        }
                    });

                    if has_derive_kernel {
                        // Extract #[kernel(source = "...", function = "...")]
                        let mut source = None;
                        let mut function = None;
                        let mut includes = Vec::new();
                        for attr in &node.attrs {
                            if attr.path().is_ident("kernel") {
                                // Parse nested meta
                                let parser = syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated;
                                if let Ok(nested) = attr.parse_args_with(parser) {
                                    for meta in nested {
                                        if let Meta::NameValue(nv) = meta {
                                            if nv.path.is_ident("source") {
                                                if let Expr::Lit(expr_lit) = nv.value {
                                                    if let Lit::Str(lit) = expr_lit.lit {
                                                        source = Some(lit.value());
                                                    }
                                                }
                                            } else if nv.path.is_ident("function") {
                                                if let Expr::Lit(expr_lit) = nv.value {
                                                    if let Lit::Str(lit) = expr_lit.lit {
                                                        function = Some(lit.value());
                                                    }
                                                }
                                            } else if nv.path.is_ident("include") {
                                                if let Expr::Array(arr) = nv.value {
                                                    for elem in arr.elems {
                                                        if let Expr::Lit(expr_lit) = elem {
                                                            if let Lit::Str(lit) = expr_lit.lit {
                                                                includes.push(lit.value());
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if let (Some(src), Some(_func)) = (source, function) {
                            self.kernels.push(FoundryKernel {
                                name: node.ident.to_string(),
                                source: src,
                                includes,
                            });
                        }
                    }
                }
            }

            let mut visitor = KernelVisitor { kernels: Vec::new() };

            for entry in walkdir::WalkDir::new(&metals_root)
                .into_iter()
                .filter_map(Result::ok)
                .filter(|e| e.file_type().is_file() && e.path().extension().map_or(false, |ext| ext == "rs"))
            {
                let content = fs::read_to_string(entry.path()).expect("read rust file");
                if let Ok(file) = parse_file(&content) {
                    visitor.visit_file(&file);
                }
            }

            // Compile found kernels
            for kernel in visitor.kernels {
                let metal_input = if kernel.includes.is_empty() {
                    metals_root.join(&kernel.source)
                } else {
                    let name = kernel.name.as_str();
                    let generated = out_dir.join(format!("{name}_bundled.metal"));
                    let mut bundle_source = String::new();

                    // Add includes
                    for include in kernel.includes {
                        let include_path = metals_root.join(&include);
                        let abs_path = fs::canonicalize(&include_path).unwrap_or_else(|err| {
                            panic!("Failed to canonicalize include '{}': {err}", include_path.display());
                        });
                        bundle_source.push_str(&format!(r#"#include "{}""#, abs_path.display()));
                        bundle_source.push('\n');
                    }

                    // Add main source
                    let source_path = metals_root.join(&kernel.source);
                    let abs_source = fs::canonicalize(&source_path).unwrap_or_else(|err| {
                        panic!("Failed to canonicalize source '{}': {err}", source_path.display());
                    });
                    bundle_source.push_str(&format!(r#"#include "{}""#, abs_source.display()));
                    bundle_source.push('\n');

                    fs::write(&generated, bundle_source).expect("failed to write bundled metal source");
                    generated
                };

                if !metal_input.exists() {
                    println!("cargo:warning=Kernel source not found: {}", metal_input.display());
                    continue;
                }

                let name = kernel.name.as_str();
                compile_metal(&metal_input, &name, &out_dir, &module_cache);
            }
        }
    } else {
        println!("Skipping metallib build (using source kernels)");
    }
}

fn compile_metal(input_path: &PathBuf, output_name: &str, out_dir: &PathBuf, module_cache: &PathBuf) {
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
    metal_cmd.env("CLANG_MODULE_CACHE_PATH", &module_cache);
    let status = metal_cmd.status().expect("run metal compiler");
    if !status.success() {
        panic!("Command failed: xcrun metal {} -> {}", input_path.display(), air.display());
    }
    // .air → .metallib
    let status = Command::new("xcrun")
        .args(&[
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
