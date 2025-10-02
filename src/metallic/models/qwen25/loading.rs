use crate::gguf::{GGUFValue, model_loader::GGUFModel};
use crate::metallic::tensor::TensorHostSlice;
use crate::metallic::{
    Context, Dtype, MetalError, Tensor, TensorElement,
    models::{LoadableModel, Qwen25, Qwen25Config},
};
use std::{any::TypeId, borrow::Cow};

fn tensor_data_as_f32<'a, T: TensorElement>(tensor: &'a Tensor<T>) -> Cow<'a, [f32]> {
    match tensor.as_slice() {
        TensorHostSlice::Borrowed(raw) => {
            if T::DTYPE == Dtype::F32 {
                debug_assert_eq!(std::mem::size_of::<T::Scalar>(), std::mem::size_of::<f32>());
                let f32_slice = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, raw.len()) };
                Cow::Borrowed(f32_slice)
            } else {
                Cow::Owned(raw.iter().copied().map(T::to_f32).collect())
            }
        }
        TensorHostSlice::Owned(owned) => Cow::Owned(owned.into_iter().map(T::to_f32).collect()),
    }
}

fn copy_f32_into_tensor<TDst: TensorElement>(src: &[f32], dst: &mut Tensor<TDst>) -> Result<(), MetalError> {
    if src.len() != dst.len() {
        return Err(MetalError::DimensionMismatch {
            expected: dst.len(),
            actual: src.len(),
        });
    }

    TDst::copy_from_f32_slice(src, dst.as_mut_slice());
    Ok(())
}

fn copy_tensor_data_into_slice<TSrc: TensorElement, TDst: TensorElement>(
    src: &Tensor<TSrc>,
    dst_slice: &mut [TDst::Scalar],
) -> Result<(), MetalError> {
    if src.len() != dst_slice.len() {
        return Err(MetalError::DimensionMismatch {
            expected: dst_slice.len(),
            actual: src.len(),
        });
    }

    if TypeId::of::<TSrc::Scalar>() == TypeId::of::<TDst::Scalar>()
        && std::mem::size_of::<TSrc::Scalar>() == std::mem::size_of::<TDst::Scalar>()
    {
        let src_slice = src.as_slice();
        unsafe {
            let typed_src = std::slice::from_raw_parts(src_slice.as_ptr() as *const TDst::Scalar, src_slice.len());
            dst_slice.copy_from_slice(typed_src);
        }
        return Ok(());
    }

    match (TSrc::DTYPE, TDst::DTYPE) {
        (Dtype::F32, _) => {
            let src_slice = src.as_slice();
            debug_assert_eq!(std::mem::size_of::<TSrc::Scalar>(), std::mem::size_of::<f32>());
            let src_f32 = unsafe { std::slice::from_raw_parts(src_slice.as_ptr() as *const f32, src_slice.len()) };
            TDst::copy_from_f32_slice(src_f32, dst_slice);
            Ok(())
        }
        (_, Dtype::F32) => {
            let src_slice = src.as_slice();
            debug_assert_eq!(std::mem::size_of::<TDst::Scalar>(), std::mem::size_of::<f32>());
            let dst_f32 = unsafe { std::slice::from_raw_parts_mut(dst_slice.as_mut_ptr() as *mut f32, dst_slice.len()) };
            for (dst, value) in dst_f32.iter_mut().zip(src_slice.iter().copied()) {
                *dst = TSrc::to_f32(value);
            }
            Ok(())
        }
        _ => {
            let src_slice = tensor_data_as_f32(src);
            TDst::copy_from_f32_slice(src_slice.as_ref(), dst_slice);
            Ok(())
        }
    }
}

fn copy_tensor_into<TSrc: TensorElement, TDst: TensorElement>(src: &Tensor<TSrc>, dst: &mut Tensor<TDst>) -> Result<(), MetalError> {
    if src.len() != dst.len() {
        return Err(MetalError::DimensionMismatch {
            expected: dst.len(),
            actual: src.len(),
        });
    }

    if let Some(layout) = tensor_matrix_layout_if_strided(dst) {
        let base_ptr = {
            let slice = dst.as_mut_slice();
            slice.as_mut_ptr()
        };

        let required = strided_required_len(layout.rows, layout.cols, layout.row_stride, layout.col_stride).ok_or_else(|| {
            MetalError::InvalidShape(format!(
                "Stride {}x{} with dims [{}x{}] overflows address space",
                layout.row_stride, layout.col_stride, layout.rows, layout.cols
            ))
        })?;

        let dst_slice = unsafe { std::slice::from_raw_parts_mut(base_ptr, required) };
        let src_f32 = tensor_data_as_f32(src);
        return copy_f32_into_strided_slice::<TDst>(
            layout.rows,
            layout.cols,
            layout.row_stride,
            layout.col_stride,
            src_f32.as_ref(),
            dst_slice,
        );
    }

    let dst_slice = dst.as_mut_slice();
    copy_tensor_data_into_slice::<TSrc, TDst>(src, dst_slice)
}

fn normalize_matrix_dims(mut dims: &[usize]) -> Option<(usize, usize)> {
    while let Some((&last, rest)) = dims.split_last() {
        if last == 1 && !rest.is_empty() {
            dims = rest;
        } else {
            break;
        }
    }

    match dims.len() {
        0 => Some((1, 1)),
        1 => Some((dims[0], 1)),
        2 => Some((dims[0], dims[1])),
        _ => None,
    }
}

fn dims_are_transposed(src_dims: &[usize], dst_dims: &[usize]) -> bool {
    if let (Some((src_rows, src_cols)), Some((dst_rows, dst_cols))) = (normalize_matrix_dims(src_dims), normalize_matrix_dims(dst_dims)) {
        src_rows == dst_cols && src_cols == dst_rows
    } else {
        false
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MatrixLayout {
    rows: usize,
    cols: usize,
    row_stride: usize,
    col_stride: usize,
}

fn tensor_matrix_layout_if_strided<T: TensorElement>(tensor: &Tensor<T>) -> Option<MatrixLayout> {
    let dims = tensor.dims();
    if dims.len() < 2 || tensor.strides.len() != dims.len() {
        return None;
    }

    let rows = dims[dims.len() - 2];
    let cols = *dims.last().unwrap_or(&1);
    let row_stride = tensor.strides[tensor.strides.len() - 2];
    let col_stride = *tensor.strides.last().unwrap_or(&1);

    let expected_row_stride = if cols == 0 { 0 } else { col_stride.saturating_mul(cols) };

    if row_stride == expected_row_stride {
        None
    } else {
        Some(MatrixLayout {
            rows,
            cols,
            row_stride,
            col_stride,
        })
    }
}

fn strided_required_len(rows: usize, cols: usize, row_stride: usize, col_stride: usize) -> Option<usize> {
    if rows == 0 || cols == 0 {
        return Some(0);
    }

    let row_extent = row_stride.checked_mul(rows - 1)?;
    let col_extent = col_stride.checked_mul(cols - 1)?;
    row_extent.checked_add(col_extent)?.checked_add(1)
}

fn copy_f32_into_strided_slice<TDst: TensorElement>(
    rows: usize,
    cols: usize,
    row_stride: usize,
    col_stride: usize,
    src: &[f32],
    dst: &mut [TDst::Scalar],
) -> Result<(), MetalError> {
    if src.len() != rows.saturating_mul(cols) {
        return Err(MetalError::DimensionMismatch {
            expected: rows.saturating_mul(cols),
            actual: src.len(),
        });
    }

    if rows == 0 || cols == 0 {
        return Ok(());
    }

    let required = strided_required_len(rows, cols, row_stride, col_stride).ok_or_else(|| {
        MetalError::InvalidShape(format!(
            "Stride {}x{} with dims [{}x{}] overflows address space",
            row_stride, col_stride, rows, cols
        ))
    })?;

    if dst.len() < required {
        return Err(MetalError::DimensionMismatch {
            expected: required,
            actual: dst.len(),
        });
    }

    for row in 0..rows {
        let src_row_offset = row * cols;
        let dst_row_offset = row * row_stride;
        for col in 0..cols {
            let dst_index = dst_row_offset + col * col_stride;
            dst[dst_index] = TDst::from_f32(src[src_row_offset + col]);
        }
    }

    Ok(())
}

fn copy_transposed_f32_into_strided_slice<TDst: TensorElement>(
    src: &[f32],
    src_rows: usize,
    src_cols: usize,
    dst: &mut [TDst::Scalar],
    layout: MatrixLayout,
) -> Result<(), MetalError> {
    if src.len() != src_rows.saturating_mul(src_cols) {
        return Err(MetalError::DimensionMismatch {
            expected: src_rows.saturating_mul(src_cols),
            actual: src.len(),
        });
    }

    if layout.rows != src_cols || layout.cols != src_rows {
        return Err(MetalError::InvalidShape(format!(
            "Destination layout {:?} is not the transpose of source dims [{}, {}]",
            layout, src_rows, src_cols
        )));
    }

    if layout.rows == 0 || layout.cols == 0 {
        return Ok(());
    }

    let required = strided_required_len(layout.rows, layout.cols, layout.row_stride, layout.col_stride).ok_or_else(|| {
        MetalError::InvalidShape(format!(
            "Stride {}x{} with dims [{}x{}] overflows address space",
            layout.row_stride, layout.col_stride, layout.rows, layout.cols
        ))
    })?;

    if dst.len() < required {
        return Err(MetalError::DimensionMismatch {
            expected: required,
            actual: dst.len(),
        });
    }

    for src_row in 0..src_rows {
        let src_row_offset = src_row * src_cols;
        for src_col in 0..src_cols {
            let dst_row = src_col;
            let dst_col = src_row;
            let dst_index = dst_row * layout.row_stride + dst_col * layout.col_stride;
            dst[dst_index] = TDst::from_f32(src[src_row_offset + src_col]);
        }
    }

    Ok(())
}

fn matrix_strides<T: TensorElement>(tensor: &Tensor<T>) -> Option<(usize, usize)> {
    if tensor.dims().len() != 2 {
        return None;
    }

    if tensor.strides.len() >= 2 {
        Some((tensor.strides[0], tensor.strides[1]))
    } else {
        let cols = tensor.dims()[1];
        Some((cols, 1))
    }
}

fn is_strided_matrix<T: TensorElement>(tensor: &Tensor<T>) -> bool {
    if let Some((row_stride, col_stride)) = matrix_strides(tensor) {
        if tensor.dims().len() != 2 {
            return false;
        }
        let cols = tensor.dims()[1];
        row_stride != cols || col_stride != 1
    } else {
        false
    }
}

fn copy_tensor_into_strided_matrix<TSrc: TensorElement, TDst: TensorElement>(
    src: &Tensor<TSrc>,
    dst: &mut Tensor<TDst>,
) -> Result<(), MetalError> {
    let dst_dims = dst.dims().to_vec();
    if dst_dims.len() != 2 {
        return copy_tensor_into(src, dst);
    }

    if src.dims() != dst_dims.as_slice() {
        return Err(MetalError::InvalidShape(format!(
            "Source dims {:?} do not match destination dims {:?} for strided copy",
            src.dims(),
            dst_dims
        )));
    }

    let rows = dst_dims[0];
    let cols = dst_dims[1];
    let src_slice = tensor_data_as_f32(src);

    if src_slice.len() != rows * cols {
        return Err(MetalError::DimensionMismatch {
            expected: rows * cols,
            actual: src_slice.len(),
        });
    }

    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let mut dst_row = dst.slice(&[row..row + 1])?;
        TDst::copy_from_f32_slice(&src_slice[start..end], dst_row.as_mut_slice());
    }

    Ok(())
}

fn copy_transposed_f32_into_slice<TDst: TensorElement>(
    src_slice: &[f32],
    src_dims: &[usize],
    dst_slice: &mut [TDst::Scalar],
    dst_dims: &[usize],
) -> Result<(), MetalError> {
    if src_dims.len() != 2 {
        return Err(MetalError::InvalidShape(format!(
            "Expected 2D source tensor for transpose, got {:?}",
            src_dims
        )));
    }

    if dst_dims.len() != 2 {
        return Err(MetalError::InvalidShape(format!(
            "Expected 2D destination tensor for transpose, got {:?}",
            dst_dims
        )));
    }

    let src_rows = src_dims[0];
    let src_cols = src_dims[1];

    if dst_dims[0] != src_cols || dst_dims[1] != src_rows {
        return Err(MetalError::InvalidShape(format!(
            "Destination dims {:?} are not the transpose of source dims {:?}",
            dst_dims, src_dims
        )));
    }

    if src_slice.len() != src_rows * src_cols {
        return Err(MetalError::DimensionMismatch {
            expected: src_rows * src_cols,
            actual: src_slice.len(),
        });
    }

    if dst_slice.len() != src_slice.len() {
        return Err(MetalError::DimensionMismatch {
            expected: src_slice.len(),
            actual: dst_slice.len(),
        });
    }

    let dst_cols = dst_dims[1];

    for row in 0..src_rows {
        for col in 0..src_cols {
            let src_index = row * src_cols + col;
            let dst_row = col;
            let dst_col = row;
            let dst_index = dst_row * dst_cols + dst_col;
            dst_slice[dst_index] = TDst::from_f32(src_slice[src_index]);
        }
    }

    Ok(())
}

fn copy_transposed_f32_into_strided_slice<TDst: TensorElement>(
    src_slice: &[f32],
    src_dims: &[usize],
    dst: &mut Tensor<TDst>,
    dst_dims: &[usize],
) -> Result<(), MetalError> {
    if src_dims.len() != 2 {
        return Err(MetalError::InvalidShape(format!(
            "Expected 2D source tensor for transpose, got {:?}",
            src_dims
        )));
    }

    if dst_dims.len() != 2 {
        return Err(MetalError::InvalidShape(format!(
            "Expected 2D destination tensor for transpose, got {:?}",
            dst_dims
        )));
    }

    let src_rows = src_dims[0];
    let src_cols = src_dims[1];
    let dst_rows = dst_dims[0];
    let dst_cols = dst_dims[1];

    if dst_rows != src_cols || dst_cols != src_rows {
        return Err(MetalError::InvalidShape(format!(
            "Destination dims {:?} are not the transpose of source dims {:?}",
            dst_dims, src_dims
        )));
    }

    if src_slice.len() != src_rows * src_cols {
        return Err(MetalError::DimensionMismatch {
            expected: src_rows * src_cols,
            actual: src_slice.len(),
        });
    }

    #[cfg(debug_assertions)]
    {
        let strides = matrix_strides(dst);
        if let Some((row_stride, col_stride)) = strides {
            eprintln!(
                "[qwen25::loading] copy_transposed_f32_into_strided_slice src_dims={:?} dst_dims={:?} row_stride={} col_stride={}",
                src_dims, dst_dims, row_stride, col_stride
            );
        } else {
            eprintln!(
                "[qwen25::loading] copy_transposed_f32_into_strided_slice src_dims={:?} dst_dims={:?}",
                src_dims, dst_dims
            );
        }
    }

    let mut transposed = vec![0.0f32; dst_rows * dst_cols];
    for src_row in 0..src_rows {
        for src_col in 0..src_cols {
            let src_index = src_row * src_cols + src_col;
            let dst_row = src_col;
            let dst_col = src_row;
            let dst_index = dst_row * dst_cols + dst_col;
            transposed[dst_index] = src_slice[src_index];
        }
    }

    for row in 0..dst_rows {
        let start = row * dst_cols;
        let end = start + dst_cols;
        let mut dst_row = dst.slice(&[row..row + 1])?;
        TDst::copy_from_f32_slice(&transposed[start..end], dst_row.as_mut_slice());
    }

    Ok(())
}

fn copy_tensor_transposed_into<TSrc: TensorElement, TDst: TensorElement>(
    src: &Tensor<TSrc>,
    dst: &mut Tensor<TDst>,
) -> Result<(), MetalError> {
    if src.len() != dst.len() {
        return Err(MetalError::DimensionMismatch {
            expected: dst.len(),
            actual: src.len(),
        });
    }

    if let Some(layout) = tensor_matrix_layout_if_strided(dst) {
        let base_ptr = {
            let slice = dst.as_mut_slice();
            slice.as_mut_ptr()
        };

        let dst_dims = dst.dims().to_vec();
        let Some((dst_rows, dst_cols)) = normalize_matrix_dims(&dst_dims) else {
            return Err(MetalError::InvalidShape(format!(
                "Unable to interpret destination dims {:?} as a matrix",
                dst_dims
            )));
        };

        if dst_rows != layout.rows || dst_cols != layout.cols {
            return Err(MetalError::InvalidShape(format!(
                "Layout {:?} does not match destination dims {:?}",
                layout, dst_dims
            )));
        }

        let required = strided_required_len(layout.rows, layout.cols, layout.row_stride, layout.col_stride).ok_or_else(|| {
            MetalError::InvalidShape(format!(
                "Stride {}x{} with dims [{}x{}] overflows address space",
                layout.row_stride, layout.col_stride, layout.rows, layout.cols
            ))
        })?;

        let dst_slice = unsafe { std::slice::from_raw_parts_mut(base_ptr, required) };
        let src_dims = src.dims().to_vec();
        let Some((src_rows, src_cols)) = normalize_matrix_dims(&src_dims) else {
            return Err(MetalError::InvalidShape(format!(
                "Unable to interpret source dims {:?} as a matrix",
                src_dims
            )));
        };

        let src_slice = tensor_data_as_f32(src);
        return copy_transposed_f32_into_strided_slice::<TDst>(src_slice.as_ref(), src_rows, src_cols, dst_slice, layout);
    }

    let src_slice = tensor_data_as_f32(src);
    let src_dims = src.dims().to_vec();
    let dst_dims = dst.dims().to_vec();
    let dst_is_strided = is_strided_matrix(dst);

    #[cfg(debug_assertions)]
    {
        let strides = matrix_strides(dst);
        if let Some((row_stride, col_stride)) = strides {
            eprintln!(
                "[qwen25::loading] copy_tensor_transposed_into src_dims={:?} dst_dims={:?} row_stride={} col_stride={} strided={}",
                src_dims, dst_dims, row_stride, col_stride, dst_is_strided
            );
        } else {
            eprintln!(
                "[qwen25::loading] copy_tensor_transposed_into src_dims={:?} dst_dims={:?} strided={}",
                src_dims, dst_dims, dst_is_strided
            );
        }
    }

    if dst_is_strided {
        copy_transposed_f32_into_strided_slice::<TDst>(src_slice.as_ref(), &src_dims, dst, &dst_dims)
    } else {
        let dst_slice = dst.as_mut_slice();
        copy_transposed_f32_into_slice::<TDst>(src_slice.as_ref(), &src_dims, dst_slice, &dst_dims)
    }
}

fn copy_tensor_with_optional_transpose<TSrc: TensorElement, TDst: TensorElement>(
    src: &Tensor<TSrc>,
    dst: &mut Tensor<TDst>,
) -> Result<(), MetalError> {
    let src_dims = src.dims().to_vec();
    let dst_dims = dst.dims().to_vec();
    let dst_is_strided = is_strided_matrix(dst);
    #[cfg(debug_assertions)]
    {
        let strides = matrix_strides(dst);
        if let Some((row_stride, col_stride)) = strides {
            eprintln!(
                "[qwen25::loading] copy_tensor_with_optional_transpose src_dims={:?} dst_dims={:?} row_stride={} col_stride={} strided={}",
                src_dims, dst_dims, row_stride, col_stride, dst_is_strided
            );
        } else {
            eprintln!(
                "[qwen25::loading] copy_tensor_with_optional_transpose src_dims={:?} dst_dims={:?}",
                src_dims, dst_dims
            );
        }
    }
    if src_dims == dst_dims {
        if dst_is_strided {
            copy_tensor_into_strided_matrix(src, dst)
        } else {
            copy_tensor_into(src, dst)
        }
    } else if dims_are_transposed(&src_dims, &dst_dims) {
        copy_tensor_transposed_into(src, dst)
    } else {
        if dst_is_strided {
            copy_tensor_into_strided_matrix(src, dst)
        } else {
            copy_tensor_into(src, dst)
        }
    }
}

fn pack_weight_transposed_into_fused_slice<TDst: TensorElement>(
    src_slice: &[f32],
    src_dims: &[usize],
    dst_slice: &mut [TDst::Scalar],
    dst_dims: &[usize],
    dst_col_offset: usize,
) -> Result<(), MetalError> {
    if src_dims.len() != 2 {
        return Err(MetalError::InvalidShape(format!("Expected 2D weight tensor, got {:?}", src_dims)));
    }

    if dst_dims.len() != 2 {
        return Err(MetalError::InvalidShape(format!("Fused weight must be 2D, got {:?}", dst_dims)));
    }

    let fused_rows = dst_dims[0];
    let fused_cols = dst_dims[1];

    let (out_features, in_features) = if src_dims[1] == fused_rows {
        (src_dims[0], src_dims[1])
    } else if src_dims[0] == fused_rows {
        (src_dims[1], src_dims[0])
    } else {
        return Err(MetalError::InvalidShape(format!(
            "Unable to map weight {:?} into fused layout with {} rows",
            src_dims, fused_rows
        )));
    };

    if dst_col_offset + out_features > fused_cols {
        return Err(MetalError::InvalidShape(format!(
            "Column range [{}..{}) exceeds fused weight width {}",
            dst_col_offset,
            dst_col_offset + out_features,
            fused_cols
        )));
    }

    for out_idx in 0..out_features {
        for in_idx in 0..in_features {
            let src_index = out_idx * in_features + in_idx;
            let dst_row = in_idx;
            let dst_col = dst_col_offset + out_idx;
            let dst_index = dst_row * fused_cols + dst_col;
            dst_slice[dst_index] = TDst::from_f32(src_slice[src_index]);
        }
    }

    Ok(())
}

fn copy_weight_transposed_into_fused<TSrc: TensorElement, TDst: TensorElement>(
    src: &Tensor<TSrc>,
    dst: &mut Tensor<TDst>,
    dst_col_offset: usize,
) -> Result<(), MetalError> {
    let dst_dims = dst.dims().to_vec();
    let src_slice = tensor_data_as_f32(src);
    pack_weight_transposed_into_fused_slice::<TDst>(src_slice.as_ref(), src.dims(), dst.as_mut_slice(), &dst_dims, dst_col_offset)
}

fn copy_fused_gate_up_weight<TSrc: TensorElement, TDst: TensorElement>(
    src: &Tensor<TSrc>,
    dst: &mut Tensor<TDst>,
) -> Result<(), MetalError> {
    if src.len() != dst.len() {
        return Err(MetalError::DimensionMismatch {
            expected: dst.len(),
            actual: src.len(),
        });
    }

    let src_dims = src.dims().to_vec();
    let dst_dims = dst.dims().to_vec();

    let Some((src_rows, src_cols)) = normalize_matrix_dims(&src_dims) else {
        return Err(MetalError::InvalidShape(format!(
            "Unable to interpret fused gate/up source dims {:?} as a matrix",
            src_dims
        )));
    };

    let Some((dst_rows, dst_cols)) = normalize_matrix_dims(&dst_dims) else {
        return Err(MetalError::InvalidShape(format!(
            "Unable to interpret fused gate/up destination dims {:?} as a matrix",
            dst_dims
        )));
    };

    if src_rows == dst_rows && src_cols == dst_cols {
        return copy_tensor_into(src, dst);
    }

    if src_rows == dst_cols && src_cols == dst_rows {
        let src_slice = tensor_data_as_f32(src);
        return pack_weight_transposed_into_fused_slice::<TDst>(src_slice.as_ref(), &src_dims, dst.as_mut_slice(), &dst_dims, 0);
    }

    Err(MetalError::InvalidShape(format!(
        "Unable to map fused gate/up weight with source dims {:?} into destination dims {:?}",
        src_dims, dst_dims
    )))
}

fn copy_bias_into_fused<TSrc: TensorElement, TDst: TensorElement>(
    src: &Tensor<TSrc>,
    dst: &mut Tensor<TDst>,
    dst_offset: usize,
) -> Result<(), MetalError> {
    if dst_offset + src.len() > dst.len() {
        return Err(MetalError::DimensionMismatch {
            expected: dst.len(),
            actual: dst_offset + src.len(),
        });
    }

    let end = dst_offset + src.len();
    let dst_slice = &mut dst.as_mut_slice()[dst_offset..end];
    copy_tensor_data_into_slice::<TSrc, TDst>(src, dst_slice)
}

fn parse_layer_index(name: &str) -> Option<usize> {
    let patterns = ["layers.", "layer.", "blocks.", "block.", "layer_", "layers_"];
    let lname = name.to_lowercase();
    for pat in patterns.iter() {
        if let Some(pos) = lname.find(pat) {
            let start = pos + pat.len();
            let mut digits = String::new();
            for ch in lname[start..].chars() {
                if ch.is_ascii_digit() {
                    digits.push(ch);
                } else {
                    break;
                }
            }
            if !digits.is_empty()
                && let Ok(idx) = digits.parse::<usize>()
            {
                return Some(idx);
            }
        }
    }
    let mut found = String::new();
    for ch in lname.chars() {
        if ch.is_ascii_digit() {
            found.push(ch);
        } else if !found.is_empty() {
            break;
        }
    }
    if !found.is_empty()
        && let Ok(idx) = found.parse::<usize>()
    {
        return Some(idx);
    }
    None
}

fn load_tensor_into_model<T: TensorElement>(lname: &str, tensor: &Tensor<T>, qwen: &mut Qwen25<T>) -> Result<(), MetalError> {
    // Embedding weight (token embeddings)
    if ((lname.contains("token") && lname.contains("emb")) || lname.contains("tok_emb") || lname.contains("tokembedding"))
        && tensor.len() == qwen.embed_weight.len()
    {
        copy_tensor_into(tensor, &mut qwen.embed_weight)?;
        return Ok(());
    }

    // Output / lm_head weights
    if (lname.contains("lm_head") || lname.contains("lmhead") || (lname.contains("output") && lname.contains("weight")))
        && tensor.len() == qwen.output_weight.len()
    {
        copy_tensor_with_optional_transpose(tensor, &mut qwen.output_weight)?;
        return Ok(());
    }

    // Final normalization gamma
    if lname.contains("output_norm")
        || lname.contains("final_norm")
        || lname == "norm.weight" && tensor.len() == qwen.final_norm_gamma.len()
    {
        copy_tensor_into(tensor, &mut qwen.final_norm_gamma)?;
        return Ok(());
    }

    // Weight tying from token embedding to output weight if still unset
    if lname.contains("token_embd") && lname.contains("weight") {
        let output_slice = qwen.output_weight.as_slice();
        let is_output_unset = output_slice.iter().all(|&x| T::to_f32(x) == 0.0);

        if is_output_unset && tensor.len() == qwen.output_weight.len() {
            let src_dims = tensor.dims().to_vec();
            let dst_dims = qwen.output_weight.dims().to_vec();
            if src_dims == dst_dims {
                copy_tensor_into(tensor, &mut qwen.output_weight)?;
            } else if dims_are_transposed(&src_dims, &dst_dims) {
                copy_tensor_transposed_into(tensor, &mut qwen.output_weight)?;
            } else {
                return Err(MetalError::InvalidOperation(format!(
                    "MAPPING -> token_embd.weight shape mismatch: {:?} vs {:?}",
                    src_dims, dst_dims
                )));
            }
        } else if is_output_unset {
            return Err(MetalError::InvalidOperation(format!(
                "MAPPING -> token_embd.weight size mismatch: {} vs {}",
                tensor.len(),
                qwen.output_weight.len()
            )));
        }
        return Ok(());
    }

    if let Some(layer_idx) = parse_layer_index(lname) {
        if layer_idx >= qwen.blocks.len() {
            return Ok(());
        }

        let block = &mut qwen.blocks[layer_idx];

        let d_model_layer = qwen.config.d_model;
        let kv_dim = block.kv_dim;
        let q_offset = 0;
        let k_offset = d_model_layer;
        let v_offset = d_model_layer + kv_dim;

        if (lname.contains("wq")
            || lname.contains("attn.q")
            || lname.contains("attn_q")
            || lname.contains("q_proj.weight")
            || lname.contains("query.weight")
            || lname.contains("q.weight")
            || lname.contains("attention.query.weight"))
            && tensor.len() == d_model_layer * d_model_layer
        {
            copy_weight_transposed_into_fused(tensor, &mut block.attn_qkv_weight, q_offset)?;
            return Ok(());
        }

        if (lname.contains("wk")
            || lname.contains("attn.k")
            || lname.contains("attn_k")
            || lname.contains("k_proj.weight")
            || lname.contains("key.weight")
            || lname.contains("k.weight")
            || lname.contains("attention.key.weight"))
            && tensor.len() == kv_dim * d_model_layer
        {
            copy_weight_transposed_into_fused(tensor, &mut block.attn_qkv_weight, k_offset)?;
            return Ok(());
        }

        if (lname.contains("wv")
            || lname.contains("attn.v")
            || lname.contains("attn_v")
            || lname.contains("v_proj.weight")
            || lname.contains("value.weight")
            || lname.contains("v.weight")
            || lname.contains("attention.value.weight"))
            && tensor.len() == kv_dim * d_model_layer
        {
            copy_weight_transposed_into_fused(tensor, &mut block.attn_qkv_weight, v_offset)?;
            return Ok(());
        }

        if (lname.contains("wo")
            || lname.contains("attn_out")
            || lname.contains("attn.out")
            || lname.contains("out_proj")
            || lname.contains("o.weight")
            || lname.contains("out.weight")
            || lname.contains("attention.output.weight"))
            && tensor.len() == block.attn_out_weight.len()
        {
            copy_tensor_into(tensor, &mut block.attn_out_weight)?;
            return Ok(());
        }

        if (lname.contains("attn.q.bias") || lname.contains("attn_q.bias") || lname.contains("attention.query.bias"))
            && tensor.len() == d_model_layer
        {
            copy_bias_into_fused(tensor, &mut block.attn_qkv_bias, q_offset)?;
            return Ok(());
        }

        if (lname.contains("attn.k.bias") || lname.contains("attn_k.bias") || lname.contains("attention.key.bias"))
            && tensor.len() == kv_dim
        {
            copy_bias_into_fused(tensor, &mut block.attn_qkv_bias, k_offset)?;
            return Ok(());
        }

        if (lname.contains("attn.v.bias") || lname.contains("attn_v.bias") || lname.contains("attention.value.bias"))
            && tensor.len() == kv_dim
        {
            copy_bias_into_fused(tensor, &mut block.attn_qkv_bias, v_offset)?;
            return Ok(());
        }

        if ((lname.contains("fused") && lname.contains("gate") && lname.contains("up"))
            || lname.contains("gate_up_proj.weight")
            || lname.contains("gateup_proj.weight")
            || lname.contains("gate_up.weight")
            || lname.contains("gateup.weight")
            || lname.contains("ffn_gate_up"))
            && tensor.len() == block.ffn_gate_up_weight.len()
        {
            copy_fused_gate_up_weight(tensor, &mut block.ffn_gate_up_weight)?;
            return Ok(());
        }

        if (lname.contains("mlp.gate_proj.weight")
            || lname.contains("ffn.gate")
            || lname.contains("ffn_gate")
            || lname.contains("gate.weight")
            || lname.contains("wg.weight")
            || lname.contains("w1.weight"))
            && tensor.len() == block.ffn_gate.len()
        {
            copy_tensor_with_optional_transpose(tensor, &mut block.ffn_gate)?;
            return Ok(());
        }

        if (lname.contains("mlp.up_proj.weight")
            || lname.contains("ffn.up")
            || lname.contains("ffn_up")
            || lname.contains("up.weight")
            || lname.contains("w3.weight"))
            && tensor.len() == block.ffn_up.len()
        {
            copy_tensor_with_optional_transpose(tensor, &mut block.ffn_up)?;
            return Ok(());
        }

        if (lname.contains("mlp.down_proj.weight")
            || lname.contains("ffn.down")
            || lname.contains("ffn_down")
            || lname.contains("down.weight")
            || lname.contains("w2.weight")
            || lname.contains("fc2.weight")
            || lname.contains("wo.weight"))
            && tensor.len() == block.ffn_down.len()
        {
            copy_tensor_with_optional_transpose(tensor, &mut block.ffn_down)?;
            return Ok(());
        }

        if lname.contains("mlp.gate_proj.bias")
            || lname.contains("ffn.gate.bias")
            || lname.contains("ffn_gate.bias")
            || lname.contains("gate.bias")
            || lname.contains("w1.bias")
            || lname.contains("wg.bias")
        {
            if tensor.len() == block.ffn_gate_bias.len() {
                copy_tensor_into(tensor, &mut block.ffn_gate_bias)?;
            }
            return Ok(());
        }

        if lname.contains("mlp.up_proj.bias")
            || lname.contains("ffn.up.bias")
            || lname.contains("ffn_up.bias")
            || lname.contains("up.bias")
            || lname.contains("w3.bias")
        {
            if tensor.len() == block.ffn_up_bias.len() {
                copy_tensor_into(tensor, &mut block.ffn_up_bias)?;
            }
            return Ok(());
        }

        if lname.contains("mlp.down_proj.bias")
            || lname.contains("ffn.down.bias")
            || lname.contains("ffn_down.bias")
            || lname.contains("down.bias")
            || lname.contains("w2.bias")
            || lname.contains("b2.bias")
            || lname.contains("fc2.bias")
            || lname.contains("wo.bias")
        {
            if tensor.len() == block.ffn_down_bias.len() {
                copy_tensor_into(tensor, &mut block.ffn_down_bias)?;
            }
            return Ok(());
        }

        if ((lname.contains("attn_norm") && lname.contains("gamma"))
            || lname.contains("attn.g")
            || lname.contains("attn_norm.weight")
            || lname.contains("attention.layernorm.weight")
            || lname.contains("ln_attn.weight"))
            && tensor.len() == block.attn_norm_gamma.len()
        {
            copy_tensor_into(tensor, &mut block.attn_norm_gamma)?;
            return Ok(());
        }

        if ((lname.contains("proj_norm") && lname.contains("gamma"))
            || lname.contains("proj.g")
            || lname.contains("ffn_norm.weight")
            || lname.contains("ffn.layernorm.weight")
            || lname.contains("ln_ffn.weight"))
            && tensor.len() == block.ffn_norm_gamma.len()
        {
            copy_tensor_into(tensor, &mut block.ffn_norm_gamma)?;
            return Ok(());
        }
    }

    // If we reach here, tensor was not handled specifically.
    Ok(())
}

/// Implement the LoadableModel trait so Qwen25 can be created from a GGUFModel
impl<T: TensorElement> LoadableModel<T> for Qwen25<T> {
    fn load_from_gguf(gguf_model: &GGUFModel, ctx: &mut Context<T>) -> Result<Self, MetalError> {
        let cfg = extract_config_from_ggufmodel(gguf_model);

        // Instantiate Qwen25 with default-initialized weights
        let mut qwen = Qwen25::new(cfg, ctx)?;

        for (name, descriptor) in &gguf_model.tensors {
            let lname = name.to_lowercase();
            let materialized = gguf_model.materialize_tensor::<T>(name, &*ctx).map_err(|err| {
                MetalError::InvalidOperation(format!(
                    "Failed to materialize tensor '{}' (dtype={:?}): {err}",
                    name,
                    descriptor.data_type()
                ))
            })?;

            load_tensor_into_model(&lname, &materialized, &mut qwen)
                .map_err(|err| MetalError::InvalidOperation(format!("Failed to load tensor '{}' into model: {err}", name)))?;
        }

        Ok(qwen)
    }
}

fn extract_config_from_ggufmodel(gguf_model: &GGUFModel) -> Qwen25Config {
    // Build config heuristically from metadata (fallbacks kept consistent with qwen25::new)
    // Get our metadata sizes with variations and defaults
    let d_model = gguf_model.get_metadata_u32_or(
        &[
            "qwen2.d_model",
            "model.d_model",
            "qwen2.embedding_length",
            "model.hidden_size",
            "llama.embedding_length",
        ],
        896,
    ) as usize;
    let ff_dim = gguf_model.get_metadata_u32_or(
        &[
            "qwen2.ff_dim",
            "model.ff_dim",
            "qwen2.feed_forward_length",
            "model.intermediate_size",
        ],
        4864,
    ) as usize;
    let n_heads = gguf_model.get_metadata_u32_or(
        &[
            "qwen2.n_heads",
            "model.n_heads",
            "qwen2.attention.head_count",
            "model.num_attention_heads",
        ],
        14,
    ) as usize;
    let n_kv_heads = gguf_model.get_metadata_u32_or(
        &[
            "qwen2.n_kv_heads",
            "model.n_kv_heads",
            "qwen2.attention.head_count_kv",
            "model.num_key_value_heads",
        ],
        2,
    ) as usize;
    let n_layers = gguf_model.get_metadata_u32_or(
        &["qwen2.n_layers", "model.n_layers", "qwen2.block_count", "model.num_hidden_layers"],
        24,
    ) as usize;
    let seq_len = gguf_model.get_metadata_u32_or(&["qwen2.context_length", "model.context_length"], 32768) as usize;
    let rope_freq_base = gguf_model.get_metadata_f32_or(&["qwen2.rope_theta", "qwen2.rope.freq_base", "model.rope.freq_base"], 1_000_000.0);
    let rms_eps = gguf_model.get_metadata_f32_or(
        &[
            "qwen2.rms_norm_eps",
            "qwen2.rope.rms_eps",
            "qwen2.attention.layer_norm_rms_epsilon",
            "model.rms_norm_eps",
        ],
        1e-6,
    );

    // Determine vocabulary size:
    // 1) Prefer explicit `vocab_size` or `model.vocab_size` metadata
    // 2) Otherwise, if GGUF provides `tokenizer.ggml.tokens` (an array), use its length
    // 3) Fallback to legacy default (32000)
    let vocab_size = if let Some(GGUFValue::U32(v)) = gguf_model
        .metadata
        .entries
        .get("vocab_size")
        .or_else(|| gguf_model.metadata.entries.get("model.vocab_size"))
    {
        *v as usize
    } else if let Some(GGUFValue::Array(arr)) = gguf_model.metadata.entries.get("tokenizer.ggml.tokens") {
        // Use tokenizer token count when available to avoid token-id out-of-range errors.
        arr.len()
    } else {
        151936usize // vocab_size from qwen25 config.json
    };

    Qwen25Config {
        n_layers,
        d_model,
        ff_dim,
        n_heads,
        n_kv_heads,
        seq_len,
        vocab_size,
        rope_freq_base,
        rms_eps,
    }
}

#[cfg(test)]
mod tests {
    use super::super::Qwen25Config;
    use super::super::transformer_block::TransformerBlock;
    use super::{copy_tensor_with_optional_transpose, copy_transposed_f32_into_slice, pack_weight_transposed_into_fused_slice};
    use crate::metallic::tensor::TensorStorage;
    use crate::metallic::{Context, F32Element, MetalError, Tensor};

    #[test]
    fn copy_transposed_helper_handles_row_major_export() {
        let src_dims = vec![6, 4];
        let dst_dims = vec![4, 6];
        let src: Vec<f32> = (1..=24).map(|v| v as f32).collect();
        let mut dst = vec![0.0f32; dst_dims[0] * dst_dims[1]];

        copy_transposed_f32_into_slice::<F32Element>(&src, &src_dims, &mut dst, &dst_dims).unwrap();

        assert_eq!(
            dst,
            vec![
                1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 3.0, 7.0, 11.0, 15.0, 19.0, 23.0, 4.0, 8.0, 12.0, 16.0,
                20.0, 24.0,
            ],
        );
    }

    #[test]
    fn copy_transposed_helper_handles_column_major_export() {
        let src_dims = vec![4, 6];
        let dst_dims = vec![6, 4];
        let src: Vec<f32> = (1..=24).map(|v| v as f32).collect();
        let mut dst = vec![0.0f32; dst_dims[0] * dst_dims[1]];

        copy_transposed_f32_into_slice::<F32Element>(&src, &src_dims, &mut dst, &dst_dims).unwrap();

        assert_eq!(
            dst,
            vec![
                1.0, 7.0, 13.0, 19.0, 2.0, 8.0, 14.0, 20.0, 3.0, 9.0, 15.0, 21.0, 4.0, 10.0, 16.0, 22.0, 5.0, 11.0, 17.0, 23.0, 6.0, 12.0,
                18.0, 24.0,
            ],
        );
    }

    #[test]
    fn pack_weight_transposes_row_major_layouts() {
        let src_dims = vec![6, 4];
        let src: Vec<f32> = (1..=24).map(|v| v as f32).collect();
        let fused_dims = vec![4, 6];
        let mut fused = vec![0.0; fused_dims[0] * fused_dims[1]];

        pack_weight_transposed_into_fused_slice::<F32Element>(&src, &src_dims, &mut fused, &fused_dims, 0).unwrap();

        assert_eq!(
            fused,
            vec![
                1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 3.0, 7.0, 11.0, 15.0, 19.0, 23.0, 4.0, 8.0, 12.0, 16.0,
                20.0, 24.0,
            ],
        );
    }

    #[test]
    fn pack_weight_transposes_column_major_exports() {
        let src_dims = vec![4, 6];
        let src: Vec<f32> = (1..=24).map(|v| v as f32).collect();
        let fused_dims = vec![4, 6];
        let mut fused = vec![0.0; fused_dims[0] * fused_dims[1]];

        pack_weight_transposed_into_fused_slice::<F32Element>(&src, &src_dims, &mut fused, &fused_dims, 0).unwrap();

        assert_eq!(
            fused,
            vec![
                1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 3.0, 7.0, 11.0, 15.0, 19.0, 23.0, 4.0, 8.0, 12.0, 16.0,
                20.0, 24.0,
            ],
        );
    }

    #[test]
    fn pack_weight_errors_when_rows_mismatch_fused_input() {
        let src_dims = vec![5, 5];
        let src = vec![0.0; src_dims[0] * src_dims[1]];
        let fused_dims = vec![4, 6];
        let mut fused = vec![0.0; fused_dims[0] * fused_dims[1]];

        let err = pack_weight_transposed_into_fused_slice::<F32Element>(&src, &src_dims, &mut fused, &fused_dims, 0)
            .expect_err("expected invalid shape error");

        match err {
            MetalError::InvalidShape(_) => {}
            other => panic!("unexpected error: {:?}", other),
        }
    }

    fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = data[r * cols + c];
            }
        }
        out
    }

    fn build_test_config() -> Qwen25Config {
        Qwen25Config {
            n_layers: 1,
            d_model: 6,
            ff_dim: 4,
            n_heads: 2,
            n_kv_heads: 1,
            seq_len: 8,
            vocab_size: 32,
            rope_freq_base: 10000.0,
            rms_eps: 1e-5,
        }
    }

    #[test]
    fn fused_gate_up_views_preserve_independent_halves() -> Result<(), MetalError> {
        let mut ctx = Context::<F32Element>::new()?;
        let cfg = build_test_config();

        let gate_data: Vec<f32> = (0..cfg.ff_dim * cfg.d_model).map(|idx| idx as f32 + 0.5).collect();
        let up_data: Vec<f32> = (0..cfg.ff_dim * cfg.d_model).map(|idx| idx as f32 + 1000.0).collect();

        let gate_tensor = Tensor::<F32Element>::from_f32_slice(vec![cfg.ff_dim, cfg.d_model], TensorStorage::Dedicated(&ctx), &gate_data)?;
        let up_tensor = Tensor::<F32Element>::from_f32_slice(vec![cfg.ff_dim, cfg.d_model], TensorStorage::Dedicated(&ctx), &up_data)?;

        let mut block = TransformerBlock::new(&cfg, &mut ctx)?;

        copy_tensor_with_optional_transpose(&gate_tensor, &mut block.ffn_gate)?;
        let expected_gate = transpose(&gate_data, cfg.ff_dim, cfg.d_model);
        assert_eq!(block.ffn_gate.to_f32_vec(), expected_gate);

        let up_before = block.ffn_up.to_f32_vec();
        assert!(up_before.iter().all(|&v| v == 0.0));

        copy_tensor_with_optional_transpose(&up_tensor, &mut block.ffn_up)?;
        let expected_up = transpose(&up_data, cfg.ff_dim, cfg.d_model);
        assert_eq!(block.ffn_up.to_f32_vec(), expected_up);

        // Ensure the gate view remains intact after populating the up view.
        assert_eq!(block.ffn_gate.to_f32_vec(), expected_gate);

        // Validate that the fused storage reflects the independent halves.
        let fused = block.ffn_gate_up_weight.to_f32_vec();
        let mut fused_gate = Vec::with_capacity(cfg.d_model * cfg.ff_dim);
        let mut fused_up = Vec::with_capacity(cfg.d_model * cfg.ff_dim);
        for row in fused.chunks(2 * cfg.ff_dim) {
            fused_gate.extend_from_slice(&row[..cfg.ff_dim]);
            fused_up.extend_from_slice(&row[cfg.ff_dim..]);
        }
        assert_eq!(fused_gate, expected_gate);
        assert_eq!(fused_up, expected_up);

        Ok(())
    }
}
