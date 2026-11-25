//! Shared utilities for the Modern AI Course Rust/Candle port.
//!
//! Provides:
//! - [`mnist`]: Byte-for-byte compatible MNIST IDX loader with local caching.
//! - [`tensor`]: Deterministic tensor generators (`linspace`, `seeded_randn`) and
//!   `assert_allclose` helpers for `local` and `submit` test suites.

pub mod mnist;
pub mod tensor;

pub use tensor::{assert_allclose, linspace, seeded_randn};
