//! Homework 0: Setup and programming basics.
//!
//! See `README.md` for full problem descriptions.

pub mod p1_add;
pub mod p2_primes;
pub mod p3a_poly_add;
pub mod p3b_poly_mul;
pub mod p3c_poly_derivative;
pub mod polynomial;

pub use p1_add::add;
pub use p2_primes::primes;
pub use p3a_poly_add::poly_add;
pub use p3b_poly_mul::poly_mul;
pub use p3c_poly_derivative::poly_derivative;
pub use polynomial::Polynomial;
