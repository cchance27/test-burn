use std::sync::OnceLock;

use metallic_env::{EnvVarGuard, FA_PREFILL_WARPS, FoundryEnvVar};

#[test]
fn get_valid_cached_latches_first_value() {
    let _clear = EnvVarGuard::unset(FoundryEnvVar::FaPrefillWarps);
    static CACHE: OnceLock<Option<u32>> = OnceLock::new();

    {
        let _set = EnvVarGuard::set(FoundryEnvVar::FaPrefillWarps, "8");
        assert_eq!(FA_PREFILL_WARPS.get_valid_cached(&CACHE), Some(8));
    }

    {
        let _set = EnvVarGuard::set(FoundryEnvVar::FaPrefillWarps, "16");
        assert_eq!(FA_PREFILL_WARPS.get_valid_cached(&CACHE), Some(8));
    }
}
