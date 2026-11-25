//! Problem 2: Vector addition.

use candle_core::bail;
use candle_core::{Result, Tensor};

/// Add two 1D vectors `x` and `y` element-by-element without using built-in tensor addition.
///
/// Return an `Err` (e.g. via `candle_core::bail!`) if either input is not 1D or if their
/// lengths do not match.
///
/// # Inputs
/// - `x`: 1D `f32` tensor of shape `[n]`
/// - `y`: 1D `f32` tensor of shape `[n]`
///
/// # Output
/// - 1D `f32` tensor of shape `[n]` equal to `x + y`
#[allow(unused_variables)]
pub fn vector_add(x: &Tensor, y: &Tensor) -> Result<Tensor> {
    // BEGIN YOUR CODE
    // let xlen = x.dims1()?;
    // let ylen = y.dims1()?;
    // if xlen != ylen {
    //     bail!("mismatched lengths {} vs {}", xlen, ylen);
    // }
    // x.add(y)
    let mut x_vec: Vec<f32> = x.to_vec1()?;
    let y_vec: Vec<f32> = y.to_vec1()?;
    if x_vec.len() != y_vec.len() {
        bail!("mismatched lengths");
    }
    x_vec
        .iter_mut()
        .zip(y_vec.iter())
        .for_each(|(xi, yi)| *xi += yi);
    Tensor::from_vec(x_vec, x.shape(), x.device())
    // END YOUR CODE
}
