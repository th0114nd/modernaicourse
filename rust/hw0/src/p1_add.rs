//! Problem 1: Simple addition.

use std::ops::Add;

/// Add `x` and `y`.
///
/// # Inputs
/// - `x`: value of type `T`
/// - `y`: value of type `T`
///
/// # Output
/// - The sum `x + y` of type `T`.
#[allow(unused_variables)]
pub fn add<T: Add<Output = T>>(x: T, y: T) -> T {
    // BEGIN YOUR CODE
    x + y
    // END YOUR CODE
}
