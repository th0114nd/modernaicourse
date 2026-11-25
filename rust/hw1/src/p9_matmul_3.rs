//! Problem 9: Matrix-matrix multiplication approach #3 (row-by-row).

use candle_core::{Result, Tensor};

#[allow(unused_imports)]
use crate::vector_matrix_product_2;

/// Compute the matrix product `A B` by computing each row `i` as `a_i^T B` using
/// `vector_matrix_product_2`.
///
/// Return an `Err` if either input is not 2D or if `a.dim(1) != b.dim(0)`.
///
/// # Inputs
/// - `a`: 2D `f32` tensor of shape `[m, n]`
/// - `b`: 2D `f32` tensor of shape `[n, p]`
///
/// # Output
/// - 2D `f32` tensor of shape `[m, p]` equal to `A B`
#[allow(unused_variables)]
pub fn matmul_3(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
