// --- Force built_kernels always wins ---
#[cfg(feature = "built_kernels")]
#[macro_export]
macro_rules! kernel_lib {
    ($name:expr) => {
        $crate::kernels::KernelSource::Binary(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".metallib")))
    };
    ($name:expr, $($_extra:expr),+ ) => {{
        // Extras are ignored in built-kernel mode because the metallib already bundles dependencies.
        $crate::kernels::KernelSource::Binary(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".metallib")))
    }};
}

// --- Force src_kernels or debug ---
#[cfg(all(not(feature = "built_kernels"), any(debug_assertions, feature = "src_kernels")))]
#[macro_export]
macro_rules! kernel_lib {
    ($name:expr) => {
        $crate::kernels::KernelSource::Text(include_str!(concat!($name, "/kernel.metal")))
    };
    ($name:expr, $($extra:expr),+ ) => {
        $crate::kernels::KernelSource::Text(concat!(
            $(
                include_str!($extra),
                "\n"
            ),+
        ))
    };
}

// --- Default to built metallibs (release, no src_kernels, no built_kernels) ---
#[cfg(all(not(debug_assertions), not(feature = "src_kernels"), not(feature = "built_kernels")))]
#[macro_export]
macro_rules! kernel_lib {
    ($name:expr) => {
        $crate::kernels::KernelSource::Binary(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".metallib")))
    };
    ($name:expr, $($_extra:expr),+ ) => {{ $crate::kernels::KernelSource::Binary(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".metallib"))) }};
}
