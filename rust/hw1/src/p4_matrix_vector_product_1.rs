//! Problem 4: Matrix-vector product approach #1 (row inner products).

use candle_core::{bail, Result, Tensor};

use crate::vector_inner_product;

/// Compute the matrix-vector product `A x` using row inner products (`vector_inner_product`).
///
/// Return an `Err` if `a` is not 2D, `x` is not 1D, or `a.dim(1) != x.dim(0)`.
///
/// # Inputs
/// - `a`: 2D `f32` tensor of shape `[m, n]`
/// - `x`: 1D `f32` tensor of shape `[n]`
///
/// # Output
/// - 1D `f32` tensor of shape `[m]` equal to `A x`
pub fn matrix_vector_product_1(a: &Tensor, x: &Tensor) -> Result<Tensor> {
    // BEGIN YOUR CODE
    let (m, n) = a.dims2()?;
    let p = x.dims1()?;
    if n != p {
        bail!("mismatched dimensions");
    }
    let mut b = Vec::with_capacity(m);
    for i in 0..m {
        let a_i = a.get(i)?;
        b.push(vector_inner_product(&a_i, x)?);
    }
    // let b: Vec<f32> = (0..m)
    //     .map(|i| {
    //         let a_i = a.get(i)?;
    //         vector_inner_product(&a_i, x)
    //     })
    //     .collect::<Result<Vec<_>>>()?;
    Tensor::new(b, a.device())
    // END YOUR CODE
}
