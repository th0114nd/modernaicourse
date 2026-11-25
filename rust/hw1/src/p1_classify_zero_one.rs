//! Problem 1: "Classical" programming for MNIST 0/1 digit classification.

use candle_core::{Result, Tensor};

/// Classify a `28 x 28` grayscale MNIST image as either `0` or `1`.
///
/// # Input
/// - `image`: 2D `f32` tensor of shape `[28, 28]` with pixel values in `[0.0, 1.0]`
///
/// # Output
/// - `0` or `1`
#[allow(unused_variables)]
pub fn classify_zero_one(image: &Tensor) -> Result<u8> {
    // BEGIN YOUR CODE
    let saturated = image.ge(0.5)?;
    let shape = saturated.shape();
    assert_eq!(shape.dim(0)?, 28);
    assert_eq!(shape.dim(1)?, 28);
    let left = saturated.narrow(1, 0, 27)?;
    let right = saturated.narrow(1, 1, 27)?;
    let mismatches = left.ne(&right)?.sum(1)?;
    let num_multicolored: u8 = mismatches.ge(4)?.sum_all()?.to_scalar()?;
    Ok(if num_multicolored >= 5 { 0 } else { 1 })
    // END YOUR CODE
}
