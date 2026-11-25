# Homework 4: Transformers & Llama 3.2 Simplified

In this homework, you will build every component of a Transformer-based LLM from scratch in
Candle, load the pretrained weights of a simplified Llama 3.2 1B Instruct model
(`zkolter/Llama-3.2-1B-Instruct-Simplified`), and run KV-cached autoregressive text generation.

> **Honor system**: Like the original course rule allowing only `Module`, `ModuleList`,
> `Parameter`, and `Buffer` from `torch.nn`, implement your solutions using only base
> `candle_core` tensor operations (`Tensor`, `Var`, arithmetic, `matmul`, `broadcast_matmul`,
> `index_select`, `narrow`, `slice_assign`, `reshape`, `transpose`) — do **not** pull in
> `candle-nn` layers or call `Tensor::silu` / `candle_nn::ops::softmax`.

## Workflow

Each graded component lives in its own file under `src/`:

- **Provided scaffolding (do not modify)**:
  - [`src/llama.rs`](src/llama.rs) — `CausalLM` trait, `LlamaConfig` (`params.json`), HuggingFace
    Hub downloader (`download_llama_files`), PyTorch `.pth` loader (`load_pth_checkpoint` via
    `candle_core::pickle::read_all`), `load_pretrained_llama3`, and `multinomial` sampler.
  - [`src/tokenizer.rs`](src/tokenizer.rs) — `Tokenizer` trait, `LlamaTokenizer` (Tiktoken BPE
    port of Meta's `tokenizer.py`), `Message`, and `ChatFormat`.
- **Graded problems**:
  - [`src/p1_linear.rs`](src/p1_linear.rs) — `Linear`
  - [`src/p2_embedding.rs`](src/p2_embedding.rs) — `Embedding`
  - [`src/p3_silu.rs`](src/p3_silu.rs) — `silu`
  - [`src/p4_rms_norm.rs`](src/p4_rms_norm.rs) — `RMSNorm`
  - [`src/p5_self_attention.rs`](src/p5_self_attention.rs) — `self_attention`
  - [`src/p6a_multi_head_attention.rs`](src/p6a_multi_head_attention.rs) — `MultiHeadAttention`
  - [`src/p6b_multi_head_attention_kv_cache.rs`](src/p6b_multi_head_attention_kv_cache.rs) — `MultiHeadAttentionKVCache`
  - [`src/p7_gated_mlp.rs`](src/p7_gated_mlp.rs) — `GatedMLP`
  - [`src/p8_transformer_block.rs`](src/p8_transformer_block.rs) — `TransformerBlock`
  - [`src/p9_llama3_simplified.rs`](src/p9_llama3_simplified.rs) — `Llama3Simplified` (plus provided `load_llama_weights`)
  - [`src/p10_generate.rs`](src/p10_generate.rs) — `generate_with_writer` (plus provided `generate` wrapper)

Run tests from `rust/`:

```bash
# Fast, offline local sanity tests (all components + synthetic multi-layer Llama KV-cache check)
cargo test -p hw4 --test local

# Run a single test while iterating
cargo test -p hw4 --test local test_multi_head_attention_kv_cache

# Full grader-equivalent submit suite (exact hw4_tests.py inputs + causal/KV-cache edge cases)
cargo test -p hw4 --test submit
```

### Pretrained Llama 3.2 1B Integration Test & Chat Runner

To keep `cargo test -p hw4 --test local` and `submit` fast and offline (avoiding an unexpected
~4.9 GB HuggingFace download during every test run), `eval_llama3` is verified in two ways:
1. `test_eval_llama3_synthetic` (in `tests/local.rs`, runs by default) verifies `LlamaConfig`
   parsing, `load_llama_weights`, finite logits, and full-vs-KV-cache equivalence on a fast
   synthetic 2-layer Llama checkpoint.
2. `test_eval_llama3_pretrained` (in `tests/local.rs`, marked `#[ignore]`) downloads and tests
   the real `zkolter/Llama-3.2-1B-Instruct-Simplified` checkpoint:

```bash
# Optional: run the real 1B checkpoint verification (~4.9 GB download on first run)
cargo test --release -p hw4 --test local test_eval_llama3_pretrained -- --ignored
```

Once everything passes, chat with your Llama 3.2 1B Simplified implementation using the CLI example:

```bash
# Default prompt ("Why is the sky blue?"), temp=0.7, max_tokens=200
cargo run --release -p hw4 --example chat

# Custom prompt, temperature, and max_tokens
cargo run --release -p hw4 --example chat -- "Why is the sky blue?" 0.7 150
```

> **Local checkpoint tip**: If you already downloaded `consolidated.00.pth`, `params.json`, and
> `tokenizer.model` (e.g. from the Python notebook), set `LLAMA_DIR=/path/to/dir` to skip
> re-downloading.

---

## PyTorch $\leftrightarrow$ Candle Quick Reference

| PyTorch (`hw4.py`) | Candle (`hw4`) |
|---|---|
| `Parameter(torch.empty(out, in))` | `Var::zeros((out_dim, in_dim), DType::F32, device)?` |
| `self.weight[Y]` (arbitrary-rank index `Y`) | `self.weight.as_tensor().index_select(&y.flatten_all()?, 0)?.reshape(out_dims)?` |
| `X @ self.weight.T` | `x.broadcast_matmul(&self.weight.as_tensor().t()?)` |
| `K.transpose(-2, -1)` before `matmul` | `k.transpose(D::Minus2, D::Minus1)?.contiguous()?` |
| `scores + mask` (`[B, H, S_q, S_k]` + `[S_q, S_k]`) | `scores.broadcast_add(mask)?` |
| `self.k_cache[:, seq_pos:end] = K` (in-place Buffer) | `self.k_cache = self.k_cache.slice_assign(&[0..1, seq_pos..end, 0..dim], &k)?;` (`&mut self`) |
| `torch.triu(torch.full((N, N), -inf), diagonal=1)` | Build `Vec<f32>` with `0.0` for `col <= row`, `f32::NEG_INFINITY` for `col > row` |
| `torch.load("consolidated.00.pth")` | `candle_core::pickle::read_all(path)?` (wrapped in `load_pth_checkpoint`) |
| `torch.multinomial(probs, 1).item()` | `hw4::multinomial(&probs)?` |

> **Why `&mut self` for KV caching?** Candle `Tensor`s are immutable reference-counted buffers.
> Taking `&mut self` on `MultiHeadAttentionKVCache::forward`, `TransformerBlock::forward`, and
> `Llama3Simplified::forward` makes KV-cache mutation explicit in the Rust type system without
> interior-mutability (`RefCell`) overhead.
>
> **Why `.contiguous()?` after `transpose`?** Candle's `Tensor::matmul` and `Tensor::reshape`
> expect contiguous layouts after permuting axes (such as transposing `(1, 2)` to split/merge
> attention heads or transposing `(D::Minus2, D::Minus1)` for $K^T$). Calling `.contiguous()?`
> before `matmul` and `reshape` avoids stride errors.
>
> **Why `Llama3Simplified` has no GQA**: Unlike stock Llama 3.2 1B (which uses Grouped Query
> Attention and RoPE), the course's `zkolter/Llama-3.2-1B-Instruct-Simplified` checkpoint uses
> standard multi-head attention (`n_heads = 32`, `dim = 2048`, `head_dim = 64`), learned
> positional embeddings (`pos_embeddings.weight`), and `ffn_dim = dim * ffn_dim_multiplier = 8192`.

---

## Problem Walkthrough

### Question 1 — Linear Layer ([`src/p1_linear.rs`](src/p1_linear.rs))

Implement `Linear` without bias:
- Store `pub weight: Var` of shape `[out_dim, in_dim]` (`DType::F32`), initialized with `Var::zeros`.
- `forward(&self, x)` computes $X W^T$ for arbitrary input shape `[..., in_dim]` $\to$ `[..., out_dim]`.

### Question 2 — Embedding Layer ([`src/p2_embedding.rs`](src/p2_embedding.rs))

Implement `Embedding`:
- Store `pub weight: Var` of shape `[num_tokens, dim]` (`DType::F32`), initialized with `Var::zeros`.
- `forward(&self, y)` maps integer token indices `y` (`DType::U32`) of arbitrary shape `[...]`
  to dense vectors of shape `[..., dim]`.

### Question 3 — SiLU Nonlinearity ([`src/p3_silu.rs`](src/p3_silu.rs))

Implement the SiLU (Sigmoid Linear Unit) activation element-wise:

$$\text{silu}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

### Question 4 — RMSNorm ([`src/p4_rms_norm.rs`](src/p4_rms_norm.rs))

Implement Root Mean Square Layer Normalization over the last dimension (`D::Minus1`):

$$\text{RMSNorm}(X) = \frac{X}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} X_j^2 + \epsilon}} \odot w$$

- Initialize `pub weight: Var` of shape `[dim]` to all ones (`Var::ones`).
- Store `pub eps: f64`.

### Question 5 — Scaled Dot-Product Self-Attention ([`src/p5_self_attention.rs`](src/p5_self_attention.rs))

Implement `self_attention(q, k, v, mask)`:

$$\text{self\_attention}(Q, K, V, \text{mask}) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d}} + \text{mask}\right) V$$

where $d$ is the last dimension of $Q$ (and $K$). Support arbitrary leading batch/head dimensions
and rectangular `(seq_q, seq_k)` shapes (needed during KV-cached decoding). When `mask` is `None`,
compute unmasked attention.

### Question 6a — Multi-Head Attention ([`src/p6a_multi_head_attention.rs`](src/p6a_multi_head_attention.rs))

Implement `MultiHeadAttention` (no caching):
1. Project $Q = X W_q^T$, $K = X W_k^T$, $V = X W_v^T$ with `Linear` layers `wq`, `wk`, `wv`.
2. Split `dim` into `n_heads` heads of dimension `dim / n_heads`, reshaping
   `[batch, seq_len, dim]` $\to$ `[batch, n_heads, seq_len, head_dim]`.
3. Apply `self_attention(&q, &k, &v, mask)`.
4. Merge heads back to `[batch, seq_len, dim]` and project with `wp`.

### Question 6b — Multi-Head Attention with KV Cache ([`src/p6b_multi_head_attention_kv_cache.rs`](src/p6b_multi_head_attention_kv_cache.rs))

Extend multi-head attention with pre-allocated `k_cache` and `v_cache` tensors of shape
`[1, max_cache_size, dim]`, initialized to zeros:
- When `use_kv_cache == true`: compute `K, V` for the new tokens, write them into `k_cache` and
  `v_cache` at `seq_pos .. seq_pos + seq_len` via `slice_assign`, and attend over the full cached
  prefix `0 .. seq_pos + seq_len`.
- When `use_kv_cache == false`: behave identically to `MultiHeadAttention` without modifying the
  cache.

### Question 7 — Gated MLP ([`src/p7_gated_mlp.rs`](src/p7_gated_mlp.rs))

Implement Llama's SwiGLU feed-forward block with `w1` (`dim -> ffn_dim`), `w2` (`ffn_dim -> dim`),
and `w3` (`dim -> ffn_dim`):

$$\text{GatedMLP}(X) = \left(\text{silu}(X W_1^T) \odot X W_3^T\right) W_2^T$$

### Question 8 — Transformer Block ([`src/p8_transformer_block.rs`](src/p8_transformer_block.rs))

Implement a pre-norm residual Transformer block containing `attn: MultiHeadAttentionKVCache`,
`norm1: RMSNorm`, `norm2: RMSNorm` (`eps = 1e-5`), and `mlp: GatedMLP`:

$$Z = X + \text{MultiHeadAttentionKVCache}(\text{RMSNorm}_1(X))$$
$$Y = Z + \text{GatedMLP}(\text{RMSNorm}_2(Z))$$

### Question 9 — Llama3Simplified ([`src/p9_llama3_simplified.rs`](src/p9_llama3_simplified.rs))

Assemble the full language model with:
- `embedding: Embedding` (`num_tokens`, `dim`)
- `pos_embeddings: Tensor` (`[max_seq_len, dim]`, initialized to zeros)
- `layers: Vec<TransformerBlock>` (length `num_layers`, each with `max_cache_size = max_seq_len`)
- `norm: RMSNorm` (`dim`, `eps = 1e-5`)
- `output: Linear` (`dim`, `num_tokens`)
- `mask: Tensor` (`[max_seq_len, max_seq_len]` causal mask with `0.0` on/below diagonal and `-inf` strictly above)

In `forward(&mut self, tokens, seq_pos, use_kv_cache)`:
1. Add token embeddings and positional embeddings for `seq_pos .. seq_pos + seq_len`.
2. Slice the causal mask: `[seq_pos..seq_pos+seq_len, 0..seq_pos+seq_len]` when `use_kv_cache` is
   `true`, or `[0..seq_len, 0..seq_len]` when `false`.
3. Pass through all Transformer blocks, final `norm`, and `output`.

The `load_llama_weights` method (mapping checkpoint keys such as `layers.{i}.attention.wo.weight`
$\to$ `layer.attn.wp.weight`) is provided for you.

### Question 10 — Autoregressive Generation ([`src/p10_generate.rs`](src/p10_generate.rs))

Implement `generate_with_writer(model, prompt_tokens, tokenizer, temp, max_tokens, writer)`:
1. Prefill the KV cache by running `model.forward` on `prompt_tokens` (`[1, prompt_len]`) with
   `seq_pos = 0` and `use_kv_cache = true`.
2. Repeat up to `max_tokens` times:
   - Take the last position's logits, divide by `temp`, apply stable softmax, and sample
     `next_token` via `multinomial(&probs)?`.
   - Record `next_token` in `generated`.
   - If `writer` is `Some(w)`, write `tokenizer.decode(&[next_token])?` and flush `w`
     *(matching `hw4_tests.py`, write before breaking on stop tokens)*.
   - Stop if `tokenizer.is_stop_token(next_token)` or `generated.len() == max_tokens`.
   - Otherwise decode one step with `[next_token]` (`[1, 1]`), incrementing `seq_pos`.
3. Return `generated` (excluding the prompt, including any sampled stop token).

The `generate(..., verbose: bool)` wrapper delegating to `stdout` is provided for you.
