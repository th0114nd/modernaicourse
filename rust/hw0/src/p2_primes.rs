//! Problem 2: Computing primes via the Sieve of Eratosthenes.

/// Compute all prime numbers up to (but not including) `n` via the Sieve of Eratosthenes.
///
/// # Input
/// - `n`: exclusive upper bound
///
/// # Output
/// - `Vec<usize>` containing all primes strictly less than `n` in ascending order.
use std::vec::Vec;

#[allow(unused_variables)]
pub fn primes(n: usize) -> Vec<usize> {
    // BEGIN YOUR CODE
    let mut out = vec![];
    if n <= 2 {
        out
    } else {
        out.push(2);

        for probe in (3usize..n).step_by(2) {
            let mut not_prime = false;
            for prime in &out {
                if probe.is_multiple_of(*prime) {
                    not_prime = true;
                }
            }
            if !not_prime {
                out.push(probe);
            }
        }
        out
    }
    // END YOUR CODE
}
