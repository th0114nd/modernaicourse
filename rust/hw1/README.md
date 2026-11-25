# Homework 1: Introduction to Linear Algebra & Candle

This homework introduces basic linear algebra operations and how to work with tensors in
[Candle](https://github.com/huggingface/candle) (`candle_core::Tensor`).

## Conventions & Honor System

- All tensors use `DType::F32` on `Device::Cpu` unless noted otherwise.
- Every tensor function returns `candle_core::Result<...>`. **When input ranks or dimensions are
  incompatible, return an `Err`** (for example with `candle_core::bail!("...")`).
- **Honor system**: In Problems 2–11, do **not** call built-in tensor addition (`+` / `Tensor::add`)
  or matrix multiplication (`Tensor::matmul` / `broadcast_mul` reductions) to shortcut the exercise—
  use explicit loops and the specific helper function requested in each problem.

## Workflow

Each problem lives in its own file under `src/`:

- Problem 1: [`src/p1_classify_zero_one.rs`](src/p1_classify_zero_one.rs) — `classify_zero_one`
- Problem 2: [`src/p2_vector_add.rs`](src/p2_vector_add.rs) — `vector_add`
- Problem 3: [`src/p3_vector_inner_product.rs`](src/p3_vector_inner_product.rs) — `vector_inner_product`
- Problem 4: [`src/p4_matrix_vector_product_1.rs`](src/p4_matrix_vector_product_1.rs) — `matrix_vector_product_1`
- Problem 5: [`src/p5_matrix_vector_product_2.rs`](src/p5_matrix_vector_product_2.rs) — `matrix_vector_product_2`
- Problem 6: [`src/p6_vector_matrix_product_2.rs`](src/p6_vector_matrix_product_2.rs) — `vector_matrix_product_2`
- Problem 7: [`src/p7_matmul_1.rs`](src/p7_matmul_1.rs) — `matmul_1`
- Problem 8: [`src/p8_matmul_2.rs`](src/p8_matmul_2.rs) — `matmul_2`
- Problem 9: [`src/p9_matmul_3.rs`](src/p9_matmul_3.rs) — `matmul_3`
- Problem 10: [`src/p10_batch_matmul.rs`](src/p10_batch_matmul.rs) — `batch_matmul`
- Problem 11: [`src/p11_block_matmul.rs`](src/p11_block_matmul.rs) — `block_matmul` (using provided `add_matmul_44`)

Run tests from `rust/`:

```bash
cargo test -p hw1 --test local
cargo test -p hw1 --test submit
```

---

## Problem 1: "Classical" programming for digit classification

Implement [`classify_zero_one`](src/p1_classify_zero_one.rs) to manually classify a `28 x 28`
grayscale MNIST image (values in `[0.0, 1.0]`) as either `0` or `1`, without machine learning.
Think intuitively about geometric features that distinguish a `0` from a `1` (e.g., center ink vs.
horizontal spread / hollowness).

You can load the filtered 0/1 MNIST dataset in any test or scratch example via
`common::mnist::load_train_zero_one(&Device::Cpu)`.

---

## Problem 2: Vector addition

For two 1D vectors $x, y \in \mathbb{R}^n$, implement [`vector_add`](src/p2_vector_add.rs) using
explicit element-by-element iteration (no built-in tensor addition). Return `Err` if either tensor
is not 1D or their lengths differ.

---

## Problem 3: Vector inner product

For two 1D vectors $x, y \in \mathbb{R}^n$, implement
[`vector_inner_product`](src/p3_vector_inner_product.rs) computing

$$\langle x, y \rangle \equiv x^T y = \sum_{i=1}^n x_i y_i$$

as an `f32` using an explicit loop (no built-in dot/matmul). Return `Err` if ranks or lengths do
not match.

---

## Problem 4: Matrix-vector product approach #1 (row inner products)

For $A \in \mathbb{R}^{m \times n}$ and $x \in \mathbb{R}^n$, compute $Ax \in \mathbb{R}^m$ in
[`matrix_vector_product_1`](src/p4_matrix_vector_product_1.rs) where entry $i$ is the inner product
$a_i^T x$ of row $i$ of $A$ with $x$. Use only `vector_inner_product`.

---

## Problem 5: Matrix-vector product approach #2 (column linear combination)

For $A \in \mathbb{R}^{m \times n}$ and $x \in \mathbb{R}^n$, compute $Ax \in \mathbb{R}^m$ in
[`matrix_vector_product_2`](src/p5_matrix_vector_product_2.rs) as a linear combination of the
columns of $A$:

$$Ax = \sum_{j=1}^n x_j \, a_{:, j}.$$

Use `vector_add` (scalar-vector multiplication is allowed).

---

## Problem 6: Vector-matrix product approach #2 (row linear combination)

For $x \in \mathbb{R}^m$ and $A \in \mathbb{R}^{m \times n}$, compute $x^T A \in \mathbb{R}^n$
(returned as a 1D tensor) in [`vector_matrix_product_2`](src/p6_vector_matrix_product_2.rs) as a
linear combination of the rows of $A$:

$$x^T A = \sum_{i=1}^m x_i \, a_{i, :}.$$

Use `vector_add` (scalar-vector multiplication is allowed).

---

## Problem 7: Matrix-matrix multiplication approach #1 (entry inner products)

For $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$, compute $AB \in \mathbb{R}^{m \times p}$
in [`matmul_1`](src/p7_matmul_1.rs) by computing each entry $(AB)_{ij} = a_i^T b_j$ using
`vector_inner_product`.

---

## Problem 8: Matrix-matrix multiplication approach #2 (column-by-column)

Compute $AB$ in [`matmul_2`](src/p8_matmul_2.rs) by computing column $j$ of $AB$ as $A b_j$ using
`matrix_vector_product_1` or `matrix_vector_product_2`.

---

## Problem 9: Matrix-matrix multiplication approach #3 (row-by-row)

Compute $AB$ in [`matmul_3`](src/p9_matmul_3.rs) by computing row $i$ of $AB$ as $a_i^T B$ using
`vector_matrix_product_2`.

---

## Problem 10: Batch matrix multiplication

For two $d$-dimensional tensors ($d \ge 2$) $A \in \mathbb{R}^{n_1 \times \dots \times n_{d-2} \times m \times n}$
and $B \in \mathbb{R}^{n_1 \times \dots \times n_{d-2} \times n \times p}$ with identical leading
batch dimensions, implement [`batch_matmul`](src/p10_batch_matmul.rs) by iterating over all leading
batch indices and multiplying the corresponding 2D slices with one of `matmul_1`/`matmul_2`/`matmul_3`.
Return `Err` if ranks differ, rank `< 2`, batch dimensions differ, or inner matrix dimensions do not
match.

---

## Problem 11: Block matrix multiplication

For $A \in \mathbb{R}^{4m \times 4n}$ and $B \in \mathbb{R}^{4n \times 4p}$, partition $A$ and $B$
into $4 \times 4$ subblocks $A_{ik}, B_{kj} \in \mathbb{R}^{4 \times 4}$ and compute each
$4 \times 4$ output block as

$$(AB)_{ij} = \sum_{k=1}^n A_{ik} B_{kj}.$$

Implement [`block_matmul`](src/p11_block_matmul.rs) using **only** the provided helper
`add_matmul_44(&z_block, &a_block, &b_block)` (which returns `z_block + a_block @ b_block`,
matching Candle's immutable tensor model). Return `Err` if either input is not 2D, inner dimensions
do not match, or any dimension is not divisible by 4.
