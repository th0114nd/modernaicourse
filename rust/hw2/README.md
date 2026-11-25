# Homework 2: Automatic Differentiation and Linear Models

This homework has two main parts:

1. **Part I — Automatic Differentiation**: Build a minimal scalar reverse-mode automatic
   differentiation engine from scratch (`Function` implementations and `compute_gradients`).
   This is the exact principle underlying PyTorch and Candle's autograd engines.
2. **Part II — Training a Digit Classifier**: Use Candle's built-in automatic differentiation
   (`Var` and `Tensor::backward`) to train a linear classifier on MNIST with minibatch
   stochastic gradient descent (SGD), without yet building `Module` or `Optimizer` abstractions
   (those come in Homework 3).

## Workflow

Each graded item lives in its own file under `src/`:

- **Provided scaffolding (do not modify)**: [`src/autograd.rs`](src/autograd.rs) — `Function` trait, `Variable`, `Multiply`, `Negate`
- **Question 1 — Function implementations**:
  - [`src/p1a_add.rs`](src/p1a_add.rs) — `Add`
  - [`src/p1b_subtract.rs`](src/p1b_subtract.rs) — `Subtract`
  - [`src/p1c_divide.rs`](src/p1c_divide.rs) — `Divide`
  - [`src/p1d_power.rs`](src/p1d_power.rs) — `Power`
  - [`src/p1e_log.rs`](src/p1e_log.rs) — `Log`
  - [`src/p1f_exp.rs`](src/p1f_exp.rs) — `Exp`
- **Question 2 — Full backward pass**:
  - [`src/p2_compute_gradients.rs`](src/p2_compute_gradients.rs) — `compute_gradients`
- **Question 3 — Loss and error**:
  - [`src/p3a_cross_entropy_loss.rs`](src/p3a_cross_entropy_loss.rs) — `cross_entropy_loss`
  - [`src/p3b_error.rs`](src/p3b_error.rs) — `error`
- **Question 4 — Minibatch SGD**:
  - [`src/p4_train_sgd.rs`](src/p4_train_sgd.rs) — `train_sgd`

Run tests from `rust/`:

```bash
cargo test -p hw2 --test local
cargo test -p hw2 --test submit
```

Once Question 3 and Question 4 pass, train and evaluate your linear classifier on full MNIST:

```bash
# Usage: cargo run --release -p hw2 --example train_mnist -- [step_size] [epochs] [batch_size]
cargo run --release -p hw2 --example train_mnist -- 0.1 5 100
```

---

## Part I: Automatic Differentiation

At the core of automatic differentiation is constructing a **computation graph** out of a series
of functions applied to variables. In [`src/autograd.rs`](src/autograd.rs), we provide two core
abstractions:

- `Variable`: a cheaply cloneable (`Rc<RefCell<...>>`) handle to a node in the computation graph,
  exposing:
  - `.value() -> f64`: scalar numerical value of the variable
  - `.grad() -> Option<f64>` / `.set_grad(f64)`: derivative of the final output with respect to this node
  - `.parents() -> Vec<Variable>`: parent nodes used to compute this node (empty for leaf variables)
  - `.function() -> Option<Rc<dyn Function>>`: the function that created this node (or `None` for leaves)
  - `.num_children() -> usize` / `.set_num_children(usize)`: number of child nodes that depend on this node
- `Function`: a trait with two methods:
  1. `forward(&self, args: &[f64]) -> f64` — evaluates the function on scalar parent values.
  2. `backward(&self, grad: f64, args: &[f64]) -> Vec<f64>` — computes the product of the incoming
     derivative $g$ and each partial derivative with respect to the inputs:

     $$\frac{\partial f(x, y)}{\partial x} \cdot g, \;\; \frac{\partial f(x, y)}{\partial y} \cdot g$$

     `backward` **always** returns a `Vec<f64>` with one entry per input argument (length 1 for
     unary functions, length 2 for binary functions).

For example, consider the expression `(x * y + x^2) / y`, broken into intermediate nodes:

```text
a = x * y
b = x^2
c = a + b
d = c / y
```

Each operator on `Variable` (`+`, `-`, `*`, `/`, unary `-`, `.pow(d)`, `.log()`, `.exp()`) calls
`Variable::apply(fn, &[parents...])`, which evaluates `forward`, increments `num_children` on
each parent, and creates a new `Variable` storing its `function` and `parents`.

### Provided Examples

In [`src/autograd.rs`](src/autograd.rs), two complete `Function` implementations are provided as
references:

- **`Multiply`**: $f(x, y) = x y$, with $\frac{\partial f}{\partial x} = y$ and
  $\frac{\partial f}{\partial y} = x$, so `backward(grad, &[x, y])` returns `vec![y * grad, x * grad]`.
- **`Negate`**: $f(x) = -x$, with $\frac{\partial f}{\partial x} = -1$, so
  `backward(grad, &[x])` returns `vec![-grad]`.

---

### Question 1: Function Implementations

Implement `Function` (`forward` and `backward`) for the remaining six operations:

1. **`Add`** ([`src/p1a_add.rs`](src/p1a_add.rs)): $f(x, y) = x + y$
2. **`Subtract`** ([`src/p1b_subtract.rs`](src/p1b_subtract.rs)): $f(x, y) = x - y$
3. **`Divide`** ([`src/p1c_divide.rs`](src/p1c_divide.rs)): $f(x, y) = x / y$
4. **`Power`** ([`src/p1d_power.rs`](src/p1d_power.rs)): $f(x) = x^d$ for fixed constant `self.d`.
   We do not differentiate with respect to `d`. Be sure to handle `d == 0.0` cleanly ($f(x) = 1$,
   $f'(x) = 0$, even when $x = 0$).
5. **`Log`** ([`src/p1e_log.rs`](src/p1e_log.rs)): $f(x) = \ln(x)$
6. **`Exp`** ([`src/p1f_exp.rs`](src/p1f_exp.rs)): $f(x) = e^x$

Once all six are implemented, expressions like `((&x * &y + x.pow(2.0)) / &y)` automatically build
the full computation graph.

---

### Question 2: Implementing the Full Backward Pass (`compute_gradients`)

Implement [`compute_gradients`](src/p2_compute_gradients.rs) (also exposed as `var.compute_gradients()`).

This simplified algorithm assumes every variable in the graph contributes only to the single final
output being differentiated (so `num_children` counts down to `0` cleanly on a single backward pass):

1. If `var.grad()` is `None` (i.e. this is the output node being differentiated), set it to `1.0`.
2. If `var.function()` is `Some(func)` (i.e. `var` is a computed node with parents):
   1. Call `func.backward(grad, &parent_values)` to get the list of `grad * partial_derivative`
      products for each parent.
   2. For each `(parent, g)` pair:
      - Add `g` to `parent.grad()` (or set it to `g` if `parent.grad()` is currently `None`).
      - Decrement `parent.num_children()` by `1`.
      - If `parent.num_children()` reaches `0`, recursively call `compute_gradients(&parent)`.

#### Worked-Through Example

Consider `a = x * y`, `b = x^2`, `c = a + b` with `x = 3.0`, `y = 4.0`. After constructing the graph:

```text
x: value=3.0,  grad=None, parents=[],     function=None,     num_children=2
y: value=4.0,  grad=None, parents=[],     function=None,     num_children=1
a: value=12.0, grad=None, parents=[x, y], function=Multiply, num_children=1
b: value=9.0,  grad=None, parents=[x],    function=Power(2), num_children=1
c: value=21.0, grad=None, parents=[a, b], function=Add,      num_children=0
```

Calling `c.compute_gradients()`:
1. Sets `c.grad = Some(1.0)` ($\frac{\partial c}{\partial c} = 1$).
2. Calls `Add.backward(1.0, &[12.0, 9.0]) = [1.0, 1.0]`, setting `a.grad = Some(1.0)`
   (`num_children -> 0`) and `b.grad = Some(1.0)` (`num_children -> 0`).
3. Recurses on `a`: `Multiply.backward(1.0, &[3.0, 4.0]) = [4.0, 3.0]`, setting
   `x.grad = Some(4.0)` (`num_children -> 1`) and `y.grad = Some(3.0)` (`num_children -> 0`).
4. Recurses on `b`: `Power(2).backward(1.0, &[3.0]) = [6.0]`, adding to `x.grad` so
   `x.grad = Some(10.0)` (`num_children -> 0`).

Final gradients: $\frac{\partial c}{\partial x} = 10.0$ and $\frac{\partial c}{\partial y} = 3.0$.

---

## Part II: Training a Digit Classifier

In Part II, we switch from our toy scalar autograd to Candle's tensor autograd to train a linear
classifier on MNIST.

### PyTorch $\leftrightarrow$ Candle Quick Reference

| PyTorch (`hw2.py`) | Candle (`hw2`) |
|---|---|
| `torch.zeros((k, n), requires_grad=True)` | `Var::zeros((k, n), DType::F32, device)?` |
| `torch.logsumexp(y_pred, dim=1)` | `y_pred.log_sum_exp(1)?` |
| `y_pred[torch.arange(N), y]` | `y_pred.gather(&y.unsqueeze(1)?, 1)?.squeeze(1)?` |
| `y_pred.argmax(dim=1)` | `y_pred.argmax(1)?` (returns `DType::U32`) |
| `torch.split(X, batch_size)` | loop `start` by `batch_size`, `x.narrow(0, start, len)?` |
| `loss.backward(); W.data -= lr * W.grad; W.grad.zero_()` | `let grads = loss.backward()?; let g = grads.get(w.as_tensor()).unwrap(); w.set(&(w.as_tensor() - (g * lr)?)?)?;` |

Note that Candle's `loss.backward()` returns a fresh `GradStore` each call, so there is no
persistent `.grad` buffer to zero out!

---

### Question 3: Cross-Entropy Loss and Error

For predicted logits $\hat{y} \in \mathbb{R}^k$ and target class $y \in \{0, \dots, k-1\}$, the
cross-entropy loss is

$$L_{ce}(\hat{y}, y) = -\log \left( \frac{\exp(\hat{y}_y)}{\sum_{j=0}^{k-1} \exp(\hat{y}_j)} \right) = -\hat{y}_y + \log \sum_{j=0}^{k-1} \exp(\hat{y}_j).$$

1. Implement [`cross_entropy_loss`](src/p3a_cross_entropy_loss.rs): given `y_pred` (`[N, k]`, `f32`)
   and `y` (`[N]`, `u32`), return the scalar (0D) mean cross-entropy loss. Use `Tensor::log_sum_exp`
   for numerical stability and `Tensor::gather` to index $\hat{y}_y$ differentiably without loops.
2. Implement [`error`](src/p3b_error.rs): given `y_pred` (`[N, k]`, `f32`) and `y` (`[N]`, `u32`),
   return the scalar (0D) `f32` mean hard-prediction classification error rate.

---

### Question 4: (Minibatch) Stochastic Gradient Descent

Implement [`train_sgd`](src/p4_train_sgd.rs) to train a linear classifier $W \in \mathbb{R}^{k \times n}$
(initialized to zeros with $k = \max(y) + 1$):

- Loop `epochs` times over `(x, y)` in sequential contiguous chunks of size `batch_size`.
- For each minibatch `(xb, yb)`:
  - Compute predictions `xb @ W^T`.
  - Compute `cross_entropy_loss(&logits, &yb)` and call `.backward()?`.
  - Update `W <- W - step_size * dL/dW` using `Var::set`.
- Return the trained weight matrix `W` (shape `[k, n]`).

### Experimenting on MNIST

Try experimenting with `step_size`, `epochs`, and `batch_size` using the provided runner until you
achieve a test error under **8%** (the best linear classifier gets slightly under **7.5%**):

```bash
cargo run --release -p hw2 --example train_mnist -- 0.1 5 100
cargo run --release -p hw2 --example train_mnist -- 0.2 20 100
```
