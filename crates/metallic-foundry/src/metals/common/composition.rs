use crate::compound::{CompoundKernel, stages::WarpLayoutStage};

pub fn manual_output(name: &str) -> CompoundKernel {
    CompoundKernel::new(name).with_manual_output(true)
}

pub fn manual_output_row_major(name: &str, warps: u32) -> CompoundKernel {
    manual_output(name).prologue(WarpLayoutStage::row_major().with_warps(warps))
}

pub fn manual_output_canonical(name: &str, warps: u32, elems_per_thread: u32) -> CompoundKernel {
    manual_output(name).prologue(
        WarpLayoutStage::canonical()
            .with_warps(warps)
            .with_elems_per_thread(elems_per_thread),
    )
}
