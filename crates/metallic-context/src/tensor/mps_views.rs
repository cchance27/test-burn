/// A lightweight description of a tensor view that can be consumed by MPS matrix APIs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MpsMatrixBatchView {
    pub batch: usize,
    pub rows: usize,
    pub columns: usize,
    pub row_bytes: usize,
    pub matrix_bytes: usize,
}
