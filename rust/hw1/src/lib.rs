//! Homework 1: Introduction to Linear Algebra and Candle.
//!
//! See `README.md` for full problem descriptions.

pub mod p10_batch_matmul;
pub mod p11_block_matmul;
pub mod p1_classify_zero_one;
pub mod p2_vector_add;
pub mod p3_vector_inner_product;
pub mod p4_matrix_vector_product_1;
pub mod p5_matrix_vector_product_2;
pub mod p6_vector_matrix_product_2;
pub mod p7_matmul_1;
pub mod p8_matmul_2;
pub mod p9_matmul_3;

pub use p10_batch_matmul::batch_matmul;
pub use p11_block_matmul::{add_matmul_44, block_matmul};
pub use p1_classify_zero_one::classify_zero_one;
pub use p2_vector_add::vector_add;
pub use p3_vector_inner_product::vector_inner_product;
pub use p4_matrix_vector_product_1::matrix_vector_product_1;
pub use p5_matrix_vector_product_2::matrix_vector_product_2;
pub use p6_vector_matrix_product_2::vector_matrix_product_2;
pub use p7_matmul_1::matmul_1;
pub use p8_matmul_2::matmul_2;
pub use p9_matmul_3::matmul_3;
