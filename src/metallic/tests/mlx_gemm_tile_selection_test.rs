use std::time::Duration;

use crate::metallic::kernels::matmul::MatMulBackendKind;
use crate::metallic::kernels::matmul::mlx_gemm::MlxGemmBackend;
use crate::metallic::kernels::{GemmTile, gemm_kernel_symbol};
use crate::metallic::tensor::MpsMatrixBatchView;
use crate::metallic::{Context, Dtype, F16Element};

#[test]
fn heuristic_selects_non_32_tile_and_records_samples() {
    let mut ctx = Context::<F16Element>::new().expect("metal context should initialize");

    let elem_size = Dtype::F16.size_bytes();

    let left_rows = 128usize;
    let left_cols = 64usize;
    let left_row_bytes = left_cols * elem_size;
    let left_matrix_bytes = left_rows * left_row_bytes;
    let left_view = MpsMatrixBatchView {
        batch: 1,
        rows: left_rows,
        columns: left_cols,
        row_bytes: left_row_bytes,
        matrix_bytes: left_matrix_bytes,
    };

    let right_rows = left_cols;
    let right_cols = 128usize;
    let right_row_bytes = right_cols * elem_size;
    let right_matrix_bytes = right_rows * right_row_bytes;
    let right_view = MpsMatrixBatchView {
        batch: 1,
        rows: right_rows,
        columns: right_cols,
        row_bytes: right_row_bytes,
        matrix_bytes: right_matrix_bytes,
    };

    let result_rows = left_rows;
    let result_cols = right_cols;
    let result_row_bytes = result_cols * elem_size;
    let result_matrix_bytes = result_rows * result_row_bytes;
    let result_view = MpsMatrixBatchView {
        batch: 1,
        rows: result_rows,
        columns: result_cols,
        row_bytes: result_row_bytes,
        matrix_bytes: result_matrix_bytes,
    };

    let backend = MlxGemmBackend::try_new(
        &mut ctx,
        false,
        false,
        &left_view,
        &right_view,
        &result_view,
        Dtype::F16,
        Dtype::F16,
        Dtype::F16,
        None,
    )
    .expect("backend creation should succeed")
    .expect("mlx backend should be selected");

    let kernel = backend.kernel();
    assert_ne!(
        kernel.tile,
        GemmTile::Bm32Bn32Bk16Wm2Wn2,
        "tile heuristic should pick a non-32x32 configuration"
    );

    let expected_symbol =
        gemm_kernel_symbol(kernel.transpose, Dtype::F16, kernel.tile).expect("known tile should map to a compiled symbol");
    assert_eq!(
        backend.kernel_symbol().expect("backend should expose kernel symbol"),
        expected_symbol
    );

    let instrumentation = ctx.matmul_instrumentation_handle();
    instrumentation.record(MatMulBackendKind::Mlx, Duration::from_micros(1));
    let samples = ctx.take_matmul_samples();
    assert!(!samples.is_empty(), "recording should preserve matmul instrumentation samples");
}
