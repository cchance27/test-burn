use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLDevice};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};

/// Create an `MPSMatrix` view into an existing `MTLBuffer`.
///
/// # Arguments
///
/// * `buffer` - The retained Metal buffer containing f32 elements.
/// * `offset` - A byte offset into the buffer where the matrix data begins.
/// * `descriptor` - Describes the matrix layout (rows, columns, rowBytes).
pub fn mps_matrix_from_buffer(
    buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
    offset: usize,
    descriptor: &Retained<MPSMatrixDescriptor>,
) -> Retained<MPSMatrix> {
    let size = unsafe { descriptor.rowBytes() * descriptor.rows() };
    debug_assert!(
        offset + size <= buffer.length(),
        "matrix dimensions exceed buffer length"
    );
    unsafe {
        MPSMatrix::initWithBuffer_offset_descriptor(MPSMatrix::alloc(), buffer, offset, descriptor)
    }
}

/// Encodes a matrix multiplication operation to a command buffer.
///
/// # Arguments
///
/// * `op` - The `MPSMatrixMultiplication` operation to encode.
/// * `command_buffer` - The command buffer to encode the operation into.
/// * `left` - The left matrix operand.
/// * `right` - The right matrix operand.
/// * `result` - The result matrix.
pub fn encode_mps_matrix_multiplication(
    op: &Retained<MPSMatrixMultiplication>,
    command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    left: &Retained<MPSMatrix>,
    right: &Retained<MPSMatrix>,
    result: &Retained<MPSMatrix>,
) {
    unsafe {
        op.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
            command_buffer,
            left,
            right,
            result,
        )
    }
}
