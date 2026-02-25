//! Foundry/runtime-specific environment variable identifiers and descriptors.

use super::{
    EnvVar, Environment, value::{EnvVarFormatError, EnvVarParseError, TypedEnvVar}
};

/// Foundry/runtime-specific environment variables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FoundryEnvVar {
    IntermediatesShared,
    KvCacheShared,
    RopeShared,
    IgnoreEosStop,
    MaxPrefillChunk,
    PrefillChunkSize,
    KvPrefixCacheDisable,
    KvPrefixCacheEntries,
    DebugStepLog,
    DebugKernelBindings,
    DebugCompiledStepSync,
    DebugStreamPoll,
    FoundryDecodeBatchSize,
    DisableBatchedPrefill,
    DebugForwardSync,
    SystemPrompt,
    DebugChatTemplate,
    RecordCbGpuTiming,
    PolicyVariant,
    ComputeDtype,
    AccumDtype,
    TuiMode,
    FoundryPerKernelProfiling,
    DebugTokenize,
    DisableChatTemplate,
    DumpMetalSourceDir,
    MaxContextLen,
    FullContextReserve,
    KvMemoryBudgetMb,
    FoundryTrace,
    SampleCpuFallback,
    DebugSampleLogits,
    DebugSampleLogitsTopN,
    DebugSampleLogitsMaxSteps,
    GemvF16Cols8,
    GemvForceSimdSumReduce,
    DebugWorkflowOps,
    SampleThreadsPerThreadgroup,
    SamplePerThreadM,
    DebugSdpa,
    FaPrefillWarps,
    DisableFaPrefillSplitK,
    FaPrefillSplitK,
    FaDecodeWarps,
    FaDecodeKeysPerWarp,
    FaDecodeScalar,
    FaDecodeTgOut,
    SdpaForceMaterialized,
    DisableFa,
    SdpaDisableOnline,
    DebugSdpaVerbose,
    DebugSdpaVerboseAll,
    SdpaDebugOnlineCompare,
    SdpaDebugOnlineCompareMinKv,
    SdpaDebugOnlineComparePrefill,
    SdpaDebugOnlineComparePrefillMinKv,
}

impl FoundryEnvVar {
    /// Obtain the canonical environment variable key for the identifier.
    #[must_use]
    pub const fn key(self) -> &'static str {
        match self {
            FoundryEnvVar::IntermediatesShared => "METALLIC_INTERMEDIATES_SHARED",
            FoundryEnvVar::KvCacheShared => "METALLIC_KV_CACHE_SHARED",
            FoundryEnvVar::RopeShared => "METALLIC_ROPE_SHARED",
            FoundryEnvVar::IgnoreEosStop => "METALLIC_IGNORE_EOS_STOP",
            FoundryEnvVar::MaxPrefillChunk => "METALLIC_MAX_PREFILL_CHUNK",
            FoundryEnvVar::PrefillChunkSize => "METALLIC_PREFILL_CHUNK_SIZE",
            FoundryEnvVar::KvPrefixCacheDisable => "METALLIC_KV_PREFIX_CACHE_DISABLE",
            FoundryEnvVar::KvPrefixCacheEntries => "METALLIC_KV_PREFIX_CACHE_ENTRIES",
            FoundryEnvVar::DebugStepLog => "METALLIC_DEBUG_STEP_LOG",
            FoundryEnvVar::DebugKernelBindings => "METALLIC_DEBUG_KERNEL_BINDINGS",
            FoundryEnvVar::DebugCompiledStepSync => "METALLIC_DEBUG_COMPILED_STEP_SYNC",
            FoundryEnvVar::DebugStreamPoll => "METALLIC_DEBUG_STREAM_POLL",
            FoundryEnvVar::FoundryDecodeBatchSize => "METALLIC_FOUNDRY_DECODE_BATCH_SIZE",
            FoundryEnvVar::DisableBatchedPrefill => "METALLIC_DISABLE_BATCHED_PREFILL",
            FoundryEnvVar::DebugForwardSync => "METALLIC_DEBUG_FORWARD_SYNC",
            FoundryEnvVar::SystemPrompt => "METALLIC_SYSTEM_PROMPT",
            FoundryEnvVar::DebugChatTemplate => "METALLIC_DEBUG_CHAT_TEMPLATE",
            FoundryEnvVar::RecordCbGpuTiming => "METALLIC_RECORD_CB_GPU_TIMING",
            FoundryEnvVar::PolicyVariant => "METALLIC_POLICY_VARIANT",
            FoundryEnvVar::ComputeDtype => "METALLIC_COMPUTE_DTYPE",
            FoundryEnvVar::AccumDtype => "METALLIC_ACCUM_DTYPE",
            FoundryEnvVar::TuiMode => "METALLIC_TUI_MODE",
            FoundryEnvVar::FoundryPerKernelProfiling => "METALLIC_FOUNDRY_PER_KERNEL_PROFILING",
            FoundryEnvVar::DebugTokenize => "METALLIC_DEBUG_TOKENIZE",
            FoundryEnvVar::DisableChatTemplate => "METALLIC_DISABLE_CHAT_TEMPLATE",
            FoundryEnvVar::DumpMetalSourceDir => "METALLIC_DUMP_METAL_SOURCE_DIR",
            FoundryEnvVar::MaxContextLen => "METALLIC_MAX_CONTEXT_LEN",
            FoundryEnvVar::FullContextReserve => "METALLIC_FULL_CONTEXT_RESERVE",
            FoundryEnvVar::KvMemoryBudgetMb => "METALLIC_KV_MEMORY_BUDGET_MB",
            FoundryEnvVar::FoundryTrace => "METALLIC_FOUNDRY_TRACE",
            FoundryEnvVar::SampleCpuFallback => "METALLIC_SAMPLE_CPU_FALLBACK",
            FoundryEnvVar::DebugSampleLogits => "METALLIC_DEBUG_SAMPLE_LOGITS",
            FoundryEnvVar::DebugSampleLogitsTopN => "METALLIC_DEBUG_SAMPLE_LOGITS_TOPN",
            FoundryEnvVar::DebugSampleLogitsMaxSteps => "METALLIC_DEBUG_SAMPLE_LOGITS_MAX_STEPS",
            FoundryEnvVar::GemvF16Cols8 => "METALLIC_GEMV_F16_COLS8",
            FoundryEnvVar::GemvForceSimdSumReduce => "METALLIC_GEMV_FORCE_SIMD_SUM_REDUCE",
            FoundryEnvVar::DebugWorkflowOps => "METALLIC_DEBUG_WORKFLOW_OPS",
            FoundryEnvVar::SampleThreadsPerThreadgroup => "METALLIC_SAMPLE_TPTG",
            FoundryEnvVar::SamplePerThreadM => "METALLIC_SAMPLE_PER_THREAD_M",
            FoundryEnvVar::DebugSdpa => "METALLIC_DEBUG_SDPA",
            FoundryEnvVar::FaPrefillWarps => "METALLIC_FA_PREFILL_WARPS",
            FoundryEnvVar::DisableFaPrefillSplitK => "METALLIC_DISABLE_FA_PREFILL_SPLITK",
            FoundryEnvVar::FaPrefillSplitK => "METALLIC_FA_PREFILL_SPLIT_K",
            FoundryEnvVar::FaDecodeWarps => "METALLIC_FA_DECODE_WARPS",
            FoundryEnvVar::FaDecodeKeysPerWarp => "METALLIC_FA_DECODE_KEYS_PER_WARP",
            FoundryEnvVar::FaDecodeScalar => "METALLIC_FA_DECODE_SCALAR",
            FoundryEnvVar::FaDecodeTgOut => "METALLIC_FA_DECODE_TG_OUT",
            FoundryEnvVar::SdpaForceMaterialized => "METALLIC_SDPA_FORCE_MATERIALIZED",
            FoundryEnvVar::DisableFa => "METALLIC_DISABLE_FA",
            FoundryEnvVar::SdpaDisableOnline => "METALLIC_SDPA_DISABLE_ONLINE",
            FoundryEnvVar::DebugSdpaVerbose => "METALLIC_DEBUG_SDPA_VERBOSE",
            FoundryEnvVar::DebugSdpaVerboseAll => "METALLIC_DEBUG_SDPA_VERBOSE_ALL",
            FoundryEnvVar::SdpaDebugOnlineCompare => "METALLIC_SDPA_DEBUG_ONLINE_COMPARE",
            FoundryEnvVar::SdpaDebugOnlineCompareMinKv => "METALLIC_SDPA_DEBUG_ONLINE_COMPARE_MIN_KV",
            FoundryEnvVar::SdpaDebugOnlineComparePrefill => "METALLIC_SDPA_DEBUG_ONLINE_COMPARE_PREFILL",
            FoundryEnvVar::SdpaDebugOnlineComparePrefillMinKv => "METALLIC_SDPA_DEBUG_ONLINE_COMPARE_PREFILL_MIN_KV",
        }
    }

    /// Convert into the unscoped [`EnvVar`] variant.
    #[must_use]
    pub const fn into_env(self) -> EnvVar {
        EnvVar::Foundry(self)
    }
}

/// Presence-only flag check: returns true if the variable exists in the process environment.
#[must_use]
pub fn is_set(var: FoundryEnvVar) -> bool {
    Environment::get(var).is_some()
}

/// Typed descriptor for the METALLIC_IGNORE_EOS_STOP truthy flag.
pub const IGNORE_EOS_STOP: TypedEnvVar<bool> = TypedEnvVar::new(FoundryEnvVar::IgnoreEosStop.into_env(), parse_truthy_flag, format_bool);
/// Typed descriptor for the prefill max chunk size.
pub const MAX_PREFILL_CHUNK: TypedEnvVar<usize> = TypedEnvVar::new(FoundryEnvVar::MaxPrefillChunk.into_env(), parse_usize, format_usize);
/// Typed descriptor for the prefill runtime chunk size.
pub const PREFILL_CHUNK_SIZE: TypedEnvVar<usize> = TypedEnvVar::new(FoundryEnvVar::PrefillChunkSize.into_env(), parse_usize, format_usize);
/// Typed descriptor for the KV prefix cache disable flag.
pub const KV_PREFIX_CACHE_DISABLE: TypedEnvVar<bool> =
    TypedEnvVar::new(FoundryEnvVar::KvPrefixCacheDisable.into_env(), parse_truthy_flag, format_bool);
/// Typed descriptor for the KV prefix cache entry limit.
pub const KV_PREFIX_CACHE_ENTRIES: TypedEnvVar<usize> =
    TypedEnvVar::new(FoundryEnvVar::KvPrefixCacheEntries.into_env(), parse_usize, format_usize);
/// Typed descriptor for decode batch size.
pub const FOUNDRY_DECODE_BATCH_SIZE: TypedEnvVar<usize> =
    TypedEnvVar::new(FoundryEnvVar::FoundryDecodeBatchSize.into_env(), parse_usize, format_usize);
/// Typed descriptor for debug stream polling toggle.
pub const DEBUG_STREAM_POLL: TypedEnvVar<bool> =
    TypedEnvVar::new(FoundryEnvVar::DebugStreamPoll.into_env(), parse_truthy_flag, format_bool);
/// Typed descriptor for METALLIC_DEBUG_KERNEL_BINDINGS.
pub const DEBUG_KERNEL_BINDINGS: TypedEnvVar<bool> =
    TypedEnvVar::new(FoundryEnvVar::DebugKernelBindings.into_env(), parse_truthy_flag, format_bool);
/// Typed descriptor for METALLIC_SYSTEM_PROMPT.
pub const SYSTEM_PROMPT: TypedEnvVar<String> = TypedEnvVar::new(FoundryEnvVar::SystemPrompt.into_env(), parse_string, format_string);
/// Typed descriptor for METALLIC_RECORD_CB_GPU_TIMING.
pub const RECORD_CB_GPU_TIMING: TypedEnvVar<bool> =
    TypedEnvVar::new(FoundryEnvVar::RecordCbGpuTiming.into_env(), parse_truthy_flag, format_bool);
/// Typed descriptor for METALLIC_POLICY_VARIANT.
pub const POLICY_VARIANT: TypedEnvVar<String> = TypedEnvVar::new(FoundryEnvVar::PolicyVariant.into_env(), parse_string, format_string);
/// Typed descriptor for METALLIC_COMPUTE_DTYPE.
pub const COMPUTE_DTYPE: TypedEnvVar<String> = TypedEnvVar::new(FoundryEnvVar::ComputeDtype.into_env(), parse_string, format_string);
/// Typed descriptor for METALLIC_ACCUM_DTYPE.
pub const ACCUM_DTYPE: TypedEnvVar<String> = TypedEnvVar::new(FoundryEnvVar::AccumDtype.into_env(), parse_string, format_string);
/// Typed descriptor for METALLIC_FOUNDRY_PER_KERNEL_PROFILING.
pub const FOUNDRY_PER_KERNEL_PROFILING: TypedEnvVar<bool> =
    TypedEnvVar::new(FoundryEnvVar::FoundryPerKernelProfiling.into_env(), parse_truthy_flag, format_bool);
/// Typed descriptor for METALLIC_DUMP_METAL_SOURCE_DIR.
pub const DUMP_METAL_SOURCE_DIR: TypedEnvVar<String> =
    TypedEnvVar::new(FoundryEnvVar::DumpMetalSourceDir.into_env(), parse_string, format_string);
/// Typed descriptor for METALLIC_MAX_CONTEXT_LEN.
pub const MAX_CONTEXT_LEN: TypedEnvVar<usize> = TypedEnvVar::new(FoundryEnvVar::MaxContextLen.into_env(), parse_usize, format_usize);
/// Typed descriptor for METALLIC_KV_MEMORY_BUDGET_MB.
pub const KV_MEMORY_BUDGET_MB: TypedEnvVar<usize> = TypedEnvVar::new(FoundryEnvVar::KvMemoryBudgetMb.into_env(), parse_usize, format_usize);
/// Typed descriptor for METALLIC_FOUNDRY_TRACE.
pub const FOUNDRY_TRACE: TypedEnvVar<bool> = TypedEnvVar::new(FoundryEnvVar::FoundryTrace.into_env(), parse_truthy_flag, format_bool);
/// Typed descriptor for METALLIC_SAMPLE_CPU_FALLBACK.
pub const SAMPLE_CPU_FALLBACK: TypedEnvVar<bool> =
    TypedEnvVar::new(FoundryEnvVar::SampleCpuFallback.into_env(), parse_truthy_flag, format_bool);
/// Typed descriptor for METALLIC_DEBUG_SAMPLE_LOGITS.
pub const DEBUG_SAMPLE_LOGITS: TypedEnvVar<bool> =
    TypedEnvVar::new(FoundryEnvVar::DebugSampleLogits.into_env(), parse_truthy_flag, format_bool);
/// Typed descriptor for METALLIC_DEBUG_SAMPLE_LOGITS_TOPN.
pub const DEBUG_SAMPLE_LOGITS_TOPN: TypedEnvVar<usize> =
    TypedEnvVar::new(FoundryEnvVar::DebugSampleLogitsTopN.into_env(), parse_usize, format_usize);
/// Typed descriptor for METALLIC_DEBUG_SAMPLE_LOGITS_MAX_STEPS.
pub const DEBUG_SAMPLE_LOGITS_MAX_STEPS: TypedEnvVar<u32> =
    TypedEnvVar::new(FoundryEnvVar::DebugSampleLogitsMaxSteps.into_env(), parse_u32, format_u32);
/// Typed descriptor for METALLIC_GEMV_F16_COLS8.
pub const GEMV_F16_COLS8: TypedEnvVar<bool> = TypedEnvVar::new(FoundryEnvVar::GemvF16Cols8.into_env(), parse_truthy_flag, format_bool);
/// Typed descriptor for METALLIC_GEMV_FORCE_SIMD_SUM_REDUCE.
pub const GEMV_FORCE_SIMD_SUM_REDUCE: TypedEnvVar<bool> =
    TypedEnvVar::new(FoundryEnvVar::GemvForceSimdSumReduce.into_env(), parse_truthy_flag, format_bool);
/// Typed descriptor for METALLIC_SAMPLE_TPTG.
pub const SAMPLE_TPTG: TypedEnvVar<usize> =
    TypedEnvVar::new(FoundryEnvVar::SampleThreadsPerThreadgroup.into_env(), parse_usize, format_usize);
/// Typed descriptor for METALLIC_SAMPLE_PER_THREAD_M.
pub const SAMPLE_PER_THREAD_M: TypedEnvVar<u32> = TypedEnvVar::new(FoundryEnvVar::SamplePerThreadM.into_env(), parse_u32, format_u32);
/// Typed descriptor for METALLIC_FA_PREFILL_WARPS.
pub const FA_PREFILL_WARPS: TypedEnvVar<u32> = TypedEnvVar::new(FoundryEnvVar::FaPrefillWarps.into_env(), parse_u32, format_u32);
/// Typed descriptor for METALLIC_FA_PREFILL_SPLIT_K.
pub const FA_PREFILL_SPLIT_K: TypedEnvVar<u32> = TypedEnvVar::new(FoundryEnvVar::FaPrefillSplitK.into_env(), parse_u32, format_u32);
/// Typed descriptor for METALLIC_FA_DECODE_WARPS.
pub const FA_DECODE_WARPS: TypedEnvVar<u32> = TypedEnvVar::new(FoundryEnvVar::FaDecodeWarps.into_env(), parse_u32, format_u32);
/// Typed descriptor for METALLIC_FA_DECODE_KEYS_PER_WARP.
pub const FA_DECODE_KEYS_PER_WARP: TypedEnvVar<u32> =
    TypedEnvVar::new(FoundryEnvVar::FaDecodeKeysPerWarp.into_env(), parse_u32, format_u32);
/// Typed descriptor for METALLIC_FA_DECODE_SCALAR.
pub const FA_DECODE_SCALAR: TypedEnvVar<String> = TypedEnvVar::new(FoundryEnvVar::FaDecodeScalar.into_env(), parse_string, format_string);
/// Typed descriptor for METALLIC_FA_DECODE_TG_OUT.
pub const FA_DECODE_TG_OUT: TypedEnvVar<String> = TypedEnvVar::new(FoundryEnvVar::FaDecodeTgOut.into_env(), parse_string, format_string);
/// Typed descriptor for METALLIC_SDPA_DEBUG_ONLINE_COMPARE_MIN_KV.
pub const SDPA_DEBUG_ONLINE_COMPARE_MIN_KV: TypedEnvVar<u32> =
    TypedEnvVar::new(FoundryEnvVar::SdpaDebugOnlineCompareMinKv.into_env(), parse_u32, format_u32);
/// Typed descriptor for METALLIC_SDPA_DEBUG_ONLINE_COMPARE_PREFILL_MIN_KV.
pub const SDPA_DEBUG_ONLINE_COMPARE_PREFILL_MIN_KV: TypedEnvVar<u32> =
    TypedEnvVar::new(FoundryEnvVar::SdpaDebugOnlineComparePrefillMinKv.into_env(), parse_u32, format_u32);

fn parse_truthy_flag(value: &str) -> Result<bool, EnvVarParseError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(false);
    }
    let lowered = trimmed.to_ascii_lowercase();
    Ok(!matches!(lowered.as_str(), "0" | "false" | "no" | "off"))
}

fn format_bool(value: &bool) -> Result<String, EnvVarFormatError> {
    Ok(value.to_string())
}

fn parse_usize(value: &str) -> Result<usize, EnvVarParseError> {
    value
        .trim()
        .parse::<usize>()
        .map_err(|_| EnvVarParseError::new("value is not a valid usize"))
}

fn format_usize(value: &usize) -> Result<String, EnvVarFormatError> {
    Ok(value.to_string())
}

fn parse_u32(value: &str) -> Result<u32, EnvVarParseError> {
    value
        .trim()
        .parse::<u32>()
        .map_err(|_| EnvVarParseError::new("value is not a valid u32"))
}

fn format_u32(value: &u32) -> Result<String, EnvVarFormatError> {
    Ok(value.to_string())
}

fn parse_string(value: &str) -> Result<String, EnvVarParseError> {
    Ok(value.to_string())
}

// Formatter signature must match `TypedEnvVar<String>`: `fn(&String) -> Result<String, _>`.
#[allow(clippy::ptr_arg)]
fn format_string(value: &String) -> Result<String, EnvVarFormatError> {
    Ok(value.to_owned())
}
