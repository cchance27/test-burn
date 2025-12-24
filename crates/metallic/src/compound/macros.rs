/// Macro to define a Conditional Kernel enum that delegates Kernel implementation to its variants.
///
/// Use this when you have multiple implementations of a kernel and want to select one at runtime
/// but present a unified `Kernel` interface.
///
/// # Example
///
/// ```ignore
/// use crate::kernel_enum;
///
/// kernel_enum!(
///     pub enum MyKernel {
///         Cpu(CpuKernel),
///         Gpu(GpuKernel),
///     }
/// );
/// ```
#[macro_export]
macro_rules! kernel_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $EnumName:ident {
            $(
                $Variant:ident($Type:ty)
            ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone)]
        $vis enum $EnumName {
            $(
                $Variant($Type),
            )*
        }

        impl $crate::foundry::Kernel for $EnumName {
            type Id = String;
            type Args = ();

            fn function_name(&self) -> &'static str {
                match self {
                    $(
                        Self::$Variant(k) => k.function_name(),
                    )*
                }
            }

            fn source(&self) -> $crate::foundry::KernelSource {
                match self {
                    $(
                        Self::$Variant(k) => k.source(),
                    )*
                }
            }

            fn includes(&self) -> $crate::foundry::Includes {
                match self {
                    $(
                        Self::$Variant(k) => k.includes(),
                    )*
                }
            }

            fn dtype(&self) -> Option<$crate::tensor::Dtype> {
                match self {
                    $(
                        Self::$Variant(k) => k.dtype(),
                    )*
                }
            }

            fn struct_defs(&self) -> String {
                match self {
                    $(
                        Self::$Variant(k) => k.struct_defs(),
                    )*
                }
            }

            fn bind(&self, encoder: &$crate::types::ComputeCommandEncoder) {
                match self {
                    $(
                        Self::$Variant(k) => k.bind(encoder),
                    )*
                }
            }

            fn dispatch_config(&self) -> $crate::types::DispatchConfig {
                match self {
                    $(
                        Self::$Variant(k) => k.dispatch_config(),
                    )*
                }
            }

            fn as_stage(&self) -> Box<dyn $crate::compound::Stage> {
                 match self {
                    $(
                        Self::$Variant(k) => k.as_stage(),
                    )*
                }
            }
        }
    };
}
