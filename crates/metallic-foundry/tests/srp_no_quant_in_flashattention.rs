use std::path::PathBuf;

fn read(path: PathBuf) -> String {
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"))
}

#[test]
fn srp_no_quantization_logic_in_flashattention_or_sdpa() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let fa_targets = [
        root.join("src/metals/flashattention/step.rs"),
        root.join("src/metals/flashattention/stages.rs"),
    ];
    let sdpa_target = root.join("src/metals/sdpa/step.rs");

    // SRP boundary: FlashAttention/SDPA consume half* tensors; quantization belongs in policy/loader codepaths.
    // These checks intentionally look for "obvious" coupling points.
    let forbidden_fa = ["crate::policy", "MetalPolicy", "Quantization", "PolicyQ", "QuantPolicy"];
    let forbidden_sdpa = ["Quantization", "PolicyQ", "QuantPolicy", "policy::q", "policy::q8", "policy::q4"];

    for path in fa_targets {
        let src = read(path.clone());
        for needle in forbidden_fa {
            assert!(!src.contains(needle), "SRP violation: found '{needle}' in {path:?}");
        }
    }

    // SDPA contains a materialized reference path that legitimately uses `PolicyF16`.
    // We only guard against quantization-specific coupling here.
    let src = read(sdpa_target.clone());
    for needle in forbidden_sdpa {
        assert!(!src.contains(needle), "SRP violation: found '{needle}' in {sdpa_target:?}");
    }
}
