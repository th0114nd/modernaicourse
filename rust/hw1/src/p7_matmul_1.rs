//! Problem 7: Matrix-matrix multiplication approach #1 (entry inner products).

use candle_core::{Result, Tensor};

#[allow(unused_imports)]
use crate::vector_inner_product;

/// Compute the matrix product `A B` by computing each entry `(A B)_{i, j}` as the inner
/// product of row `i` of `A` and column `j` of `B` using `vector_inner_product`.
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
pub fn matmul_1(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
