# Homework 0: Setup and Programming Basics

This homework is different from the other assignments in this course, in that we don't actually
build anything related to AI or chatbots yet. Instead, this is a basic assignment meant to test
two things:

1. Getting comfortable with the Cargo workspace and two-stage (`local` / `submit`) testing workflow, and
2. Evaluating some basic background in Rust programming and math proficiency of the kind you'll
   need throughout the course.

## Workflow

Each problem lives in its own file under `src/`:

- Problem 1: [`src/p1_add.rs`](src/p1_add.rs) — `add`
- Problem 2: [`src/p2_primes.rs`](src/p2_primes.rs) — `primes`
- Problem 3 (provided type): [`src/polynomial.rs`](src/polynomial.rs) — `Polynomial` *(do not modify)*
- Problem 3a: [`src/p3a_poly_add.rs`](src/p3a_poly_add.rs) — `poly_add`
- Problem 3b: [`src/p3b_poly_mul.rs`](src/p3b_poly_mul.rs) — `poly_mul`
- Problem 3c: [`src/p3c_poly_derivative.rs`](src/p3c_poly_derivative.rs) — `poly_derivative`

Fill in the code between `// BEGIN YOUR CODE` and `// END YOUR CODE`, then test locally and
"submit" from the `rust/` directory:

```bash
# Local sanity checks (like @mugrade.local_tests)
cargo test -p hw0 --test local

# Full grader-equivalent suite (like mugrade submit)
cargo test -p hw0 --test submit
```

---

## Problem 1: Setup (via simple addition)

Implement [`add`](src/p1_add.rs), which adds together two values of any type implementing
`std::ops::Add<Output = T>` (so integers stay integers and floats stay floats):

```rust
pub fn add<T: std::ops::Add<Output = T>>(x: T, y: T) -> T
```

*(Just like in the original HW0, this one is intentionally trivial: `x + y`.)*

---

## Problem 2: Computing primes

Implement [`primes`](src/p2_primes.rs) using the
[Sieve of Eratosthenes](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes), which computes all
prime numbers **up to (but not including)** `n` by iteratively marking multiples of primes as
composite (`false`) in a boolean array, then collecting the indices that remain `true`.

```rust
pub fn primes(n: usize) -> Vec<usize>
```

---

## Problem 3: Operations on polynomials

The provided [`Polynomial`](src/polynomial.rs) struct represents a polynomial as a `Vec<f64>` of
coefficients, where `coefficients[i]` is the coefficient of the $x^i$ term. For example:

- `vec![1.0, 0.0, 4.0, 3.0]` represents $3x^3 + 4x^2 + 1$
- `vec![4.0, 3.0, 5.0]` represents $5x^2 + 3x + 4$

Any term of degree beyond the length of `coefficients` implicitly has coefficient `0.0`, and
`Polynomial::new` automatically strips trailing zeros (keeping `[0.0]` for the zero polynomial).
Do **not** modify `src/polynomial.rs`.

### Problem 3a: Polynomial addition

Implement [`poly_add`](src/p3a_poly_add.rs) to add two polynomials by adding coefficients of
matching degree. For example, for

$$p_1(x) = 3x^3 + 4x^2 + 3, \quad p_2(x) = x^2 + 5x + 5$$

their sum is

$$p_1(x) + p_2(x) = 3x^3 + 5x^2 + 5x + 8.$$

### Problem 3b: Polynomial multiplication

Implement [`poly_mul`](src/p3b_poly_mul.rs) to multiply two polynomials by multiplying every term
in `p1` with every term in `p2` and summing coefficients of equal degree. For example, for

$$p_1(x) = 3x^3 + 2x + 3, \quad p_2(x) = 2x^2 + 5$$

their product is

$$p_1(x) \cdot p_2(x) = 6x^5 + 19x^3 + 6x^2 + 10x + 15.$$

### Problem 3c: Polynomial differentiation

Implement [`poly_derivative`](src/p3c_poly_derivative.rs) to compute the derivative $p'(x)$ with
respect to $x$. Each term $c x^n$ differentiates to $n c x^{n-1}$, and any constant term
differentiates to $0$. For example, for

$$p(x) = 4x^3 + 3x + 3$$

its derivative is

$$p'(x) = 12x^2 + 3.$$
