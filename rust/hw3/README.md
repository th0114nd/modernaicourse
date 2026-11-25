# Homework 3: Training Neural Networks in Candle

In this homework, you will build and train machine learning models (a linear model, a two-layer
neural network, and an arbitrary-depth multi-layer neural network) on MNIST using Candle.
While much of this functionality exists in `candle-nn` (and PyTorch's `torch.nn`), you will
implement the core building blocks from scratch using only base `candle_core` tensor operations
and the minimal trait scaffolding in [`src/nn.rs`](src/nn.rs).

> **Honor system**: Like the original course rule against `torch.nn` / `torch.nn.functional`,
> implement your solutions using only `candle_core` primitives (`Tensor`, `Var`, `GradStore`) —
> do **not** pull in `candle-nn` pre-built layers or optimizers.

## Workflow

Each graded item lives in its own file under `src/`:

- **Provided scaffolding (do not modify)**: [`src/nn.rs`](src/nn.rs) — `Module`, `Loss`, and
  `Optimizer` traits, plus `load_mnist_dataloaders` and `train_model`
- **Part I — Linear Model & Training Infrastructure**:
  - [`src/p1_linear.rs`](src/p1_linear.rs) — `Linear`
  - [`src/p2_cross_entropy_loss.rs`](src/p2_cross_entropy_loss.rs) — `CrossEntropyLoss`
  - [`src/p3_sgd.rs`](src/p3_sgd.rs) — `SGD`
  - [`src/p4_dataloader.rs`](src/p4_dataloader.rs) — `DataLoader` & `DataLoaderIter`
  - [`src/p5_epoch.rs`](src/p5_epoch.rs) — `epoch`
  - [`src/p6_eval_linear_model.rs`](src/p6_eval_linear_model.rs) — `eval_linear_model`
- **Part II — Neural Networks**:
  - [`src/p7_two_layer_nn.rs`](src/p7_two_layer_nn.rs) — `TwoLayerNN`
  - [`src/p8_eval_two_layer_nn.rs`](src/p8_eval_two_layer_nn.rs) — `eval_two_layer_nn`
  - [`src/p9_multi_layer_nn.rs`](src/p9_multi_layer_nn.rs) — `MultiLayerNN`

Run tests from `rust/`:

```bash
# Fast unit tests (skip the full MNIST training runs while iterating in debug mode)
cargo test -p hw3 --test local -- --skip test_eval

# Full local and submit suites (use --release so MNIST CPU training runs ~15-30x faster!)
cargo test --release -p hw3 --test local
cargo test --release -p hw3 --test submit
```

> **Performance tip**: Always pass `--release` when running `eval_linear_model`,
> `eval_two_layer_nn`, or `examples/train_mnist`. In release mode on CPU, `eval_linear_model`
> finishes in ~2.5s and `eval_two_layer_nn` finishes in ~15–30s; in unoptimized debug mode,
> pure-Rust GEMM on 60,000 MNIST images is much slower.

Once your models pass tests, experiment with training different architectures on MNIST:

```bash
# Train a linear classifier (default lr=0.2, epochs=20, batch_size=100)
cargo run --release -p hw3 --example train_mnist -- linear

# Train a two-layer ReLU neural network (default 784 -> 300 -> 10, lr=0.3, epochs=20)
cargo run --release -p hw3 --example train_mnist -- two-layer 0.3 20 100

# Train a deeper multi-layer ReLU network (e.g., hidden_dims = [300, 100])
cargo run --release -p hw3 --example train_mnist -- multi 0.3 20 100 300,100
```

---

## PyTorch $\leftrightarrow$ Candle Quick Reference

| PyTorch (`hw3.py`) | Candle (`hw3`) |
|---|---|
| `class Foo(nn.Module)` | `struct Foo` + `impl Module for Foo` (`forward`, `parameters`) |
| `nn.Parameter(torch.randn(out, in) * sqrt(2/in))` | `Var::randn(0.0_f32, (2.0 / in_dim as f64).sqrt() as f32, (out_dim, in_dim), device)?` |
| `X @ self.weight.T` (works for 2D & 3D `X`) | `x.broadcast_matmul(&self.weight.as_tensor().t()?)` |
| `torch.logsumexp(logits, dim=1)` | `logits.log_sum_exp(1)?` |
| `logits[torch.arange(N), y]` | `logits.gather(&y.unsqueeze(1)?, 1)?.squeeze(1)?` |
| `torch.relu(h)` | `h.relu()?` |
| `logits.argmax(-1)` | `logits.argmax(D::Minus1)?` (returns `DType::U32`) |
| `opt.zero_grad(); loss.backward(); opt.step()` | `let grads = loss.backward()?; opt.step(&grads)?;` |
| `with torch.no_grad(): p -= lr * p.grad` | `if let Some(g) = grads.get(p.as_tensor()) { p.set(&(p.as_tensor() - (g * lr)?)?)?; }` |
| `nn.ModuleList([...])` | `Vec<Linear>` + collect `parameters()` across all elements |
| `X[start:end]` | `x.narrow(0, start, len)?` |

> **Why `broadcast_matmul`?** In Candle, `Tensor::matmul` requires matching batch ranks, whereas
> `Tensor::broadcast_matmul` broadcasts the 2D transposed weight `[in_dim, out_dim]` across any
> leading batch dimensions `[..., in_dim]` (both 2D `[B, in_dim]` and 3D `[B, T, in_dim]`).
>
> **Why no `zero_grad()`?** Candle's `loss.backward()?` returns a fresh `GradStore` on every step
> instead of accumulating into mutable `.grad` fields, and `Var::set(&new_tensor)` replaces the
> parameter tensor outside the autograd graph (no `torch.no_grad()` block needed). The `Optimizer`
> trait still provides a default no-op `zero_grad(&mut self)` for API parity.

---

## Part I — Training a Linear Model

In Part I, we re-implement the MNIST linear classifier from Homework 2 in an idiomatic modular
style (`Module`, `Loss`, `Optimizer`, `DataLoader`, and `epoch`).

### Question 1 — Linear Layer ([`src/p1_linear.rs`](src/p1_linear.rs))

Implement `Linear`:
- Store trainable weights in a public `weight: Var` field of shape `[out_dim, in_dim]` (`DType::F32`).
- Initialize `weight` with random Gaussian entries scaled by $\sqrt{2 / \text{in\_dim}}$
  (Kaiming/He initialization).
- Implement `Module::forward(&self, x)` computing $x W^T$, supporting both 2D `[batch_size, in_dim]`
  and higher-rank `[..., in_dim]` inputs (returning `[..., out_dim]`).
- Implement `Module::parameters(&self)` returning `vec![self.weight.clone()]`.

### Question 2 — Cross-Entropy Loss ([`src/p2_cross_entropy_loss.rs`](src/p2_cross_entropy_loss.rs))

Implement `Loss` for `CrossEntropyLoss`:
- Given `logits` of shape `[N, k]` (`f32`) and integer targets `y` of shape `[N]` (`u32`), return
  the scalar (0D) average cross-entropy loss:

  $$L_{ce}(\hat{y}, y) = -\hat{y}_y + \log \sum_{j=0}^{k-1} \exp(\hat{y}_j)$$

- Use `Tensor::log_sum_exp` for numerical stability on large logits and `Tensor::gather` so the
  result stays differentiable through Candle's autograd.

### Question 3 — Stochastic Gradient Descent ([`src/p3_sgd.rs`](src/p3_sgd.rs))

Implement `SGD` (`SGD::new` and `Optimizer::step`):
- Store the `Vec<Var>` of parameters and `learning_rate: f64`.
- In `step(&mut self, grads: &GradStore)`, update every parameter present in `grads` via:

  $$\theta \leftarrow \theta - \eta \nabla_\theta L$$

  using `param.set(&(param.as_tensor() - (grad * self.learning_rate)?)?)?`.
- Skip any parameter whose `grads.get(param.as_tensor())` is `None`.

### Question 4 — Data Loader ([`src/p4_dataloader.rs`](src/p4_dataloader.rs))

Implement `DataLoader` and `DataLoaderIter`:
- `DataLoader::new(x, y, batch_size)` validates `batch_size >= 1` and matching row counts along
  dimension `0`.
- `DataLoader::iter(&self)` returns a fresh `DataLoaderIter` starting at offset `0` (`IntoIterator`
  for `&DataLoader` is already wired to `iter()`, so `for batch in &loader` can be called
  repeatedly across epochs).
- `DataLoaderIter::next(&mut self)` slices sequential `(x_batch, y_batch)` minibatches using
  `Tensor::narrow(0, start, len)` and returns `None` once all examples have been yielded.

### Question 5 — Optimization Epoch ([`src/p5_epoch.rs`](src/p5_epoch.rs))

Implement `epoch(model, loader, loss_fn, opt)`:
- Iterate through all minibatches `(x, y)` in `loader`.
- Compute `logits = model.forward(&x)?` and `batch_loss = loss_fn.forward(&logits, &y)?`.
- Accumulate the **example-weighted** total loss (`batch_loss * batch_len`) and total
  misclassification count so uneven final batches are averaged accurately.
- When `opt` is `Some(opt)`, run `let grads = batch_loss.backward()?; opt.step(&grads)?;`.
  When `opt` is `None` (evaluation mode), do not update parameters.
- Return `(avg_loss, avg_error)` as `(f64, f64)`.

### Question 6 — Evaluate Linear Model ([`src/p6_eval_linear_model.rs`](src/p6_eval_linear_model.rs))

Implement `eval_linear_model() -> Result<Linear>`:
- Load MNIST via the provided `load_mnist_dataloaders(100, &Device::Cpu)?`.
- Construct `Linear::new(784, 10, &device)?` and train it with `train_model` (e.g., `lr = 0.2`,
  5–20 epochs).
- Return the trained `Linear` model achieving `< 10%` test error on MNIST.

---

## Part II — Training Neural Networks

Because `epoch` and `train_model` operate on any `Module`, swapping from a linear classifier to a
deep neural network requires only implementing a new `Module`.

### Question 7 — Two-Layer Neural Network ([`src/p7_two_layer_nn.rs`](src/p7_two_layer_nn.rs))

Implement `TwoLayerNN`, computing:

$$h(x) = W_2 \, \sigma(W_1 x)$$

where $\sigma$ is ReLU (`Tensor::relu`):
- Store two public `Linear` sublayers: `linear1` (`in_dim -> hidden_dim`) and `linear2`
  (`hidden_dim -> out_dim`).
- Implement `Module::forward` and `Module::parameters` (collecting parameters from both sublayers).

### Question 8 — Evaluate Two-Layer Neural Network ([`src/p8_eval_two_layer_nn.rs`](src/p8_eval_two_layer_nn.rs))

Implement `eval_two_layer_nn() -> Result<TwoLayerNN>`:
- Train a `TwoLayerNN` (e.g., `784 -> 300 -> 10`, `lr = 0.3`, `batch_size = 100`, 10–20 epochs)
  on MNIST using `load_mnist_dataloaders` and `train_model`.
- Return the trained model achieving `< 3%` test error on `test[:2000]` (with 20 epochs it reaches
  `< 2%` test error on full MNIST).

### Question 9 — Multi-Layer Neural Network ([`src/p9_multi_layer_nn.rs`](src/p9_multi_layer_nn.rs))

Implement `MultiLayerNN`, representing an arbitrary-depth ReLU network:

$$h(x) = W_L \, \sigma(W_{L-1} \, \sigma(\dots W_2 \, \sigma(W_1 x) \dots))$$

- Store all `Linear` layers in order in the public `linears: Vec<Linear>` field (length
  `hidden_dims.len() + 1`).
- Apply ReLU after every hidden layer, and apply **no** ReLU after the final output layer.
- Implement `Module::parameters` by collecting parameters across all layers in `self.linears`.

Finally, try experimenting with deeper networks using `cargo run --release -p hw3 --example train_mnist -- multi 0.3 20 100 300,100` to see how low you can push MNIST test error (around ~1.5% is achievable)!
