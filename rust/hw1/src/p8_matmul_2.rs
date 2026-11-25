//! Problem 8: Matrix-matrix multiplication approach #2 (column-by-column).

use candle_core::{Result, Tensor};

#[allow(unused_imports)]
use crate::{matrix_vector_product_1, matrix_vector_product_2};

/// Compute the matrix product `A B` by computing each column `j` as `A b_j` using
/// `matrix_vector_product_1` or `matrix_vector_product_2`.
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
pub fn matmul_2(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
