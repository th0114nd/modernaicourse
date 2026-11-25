# Homework 5: BPE Tokenizer from Scratch & Training an LLM on TinyStories

In this homework, you will build a Byte-Pair Encoding (BPE) tokenizer from scratch and then
assemble, pretokenize, train, and sample from a decoder-only Transformer LLM on the
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset using Candle.

> **Honor system**: Like the original course rule allowing only `Module`, `ModuleList`,
> `Parameter`, and `Buffer` from `torch.nn`, implement your solutions using only base
> `candle_core` tensor operations (`Tensor`, `Var`, arithmetic, `matmul`, `broadcast_matmul`,
> `index_select`, `gather`, `narrow`, `slice_assign`, `reshape`, `transpose`) — do **not**
> pull in `candle-nn` layers, `candle_nn::ops::softmax`, `candle_nn::loss::cross_entropy`,
> `Tensor::silu`, or `candle_nn::AdamW`.

---

## Workflow & Crate Layout

- **Provided scaffolding (do not modify)**:
  - [`src/nn.rs`](src/nn.rs) — `CausalLM` and `Optimizer` traits, `multinomial` token sampler,
    `split_keep_whitespace` (Rust counterpart to Python's `re.split(r'(?=\s)', text)`), and
    `read_char_chunk` (safe UTF-8 character-count chunk reader for `pretokenize_data`).
  - [`src/tinystories.rs`](src/tinystories.rs) — `Tokenizer` trait, `Gpt2Tokenizer`
    (`tiktoken_rs::r50k_base`, identical to `tiktoken.get_encoding("gpt2")`), `padded_vocab_size`,
    and `download_tinystories` (`hf-hub` downloader for `roneneldan/TinyStories`).
- **Part I — BPE Tokenizer from Scratch**:
  - [`src/p1_text_to_corpus.rs`](src/p1_text_to_corpus.rs) — `text_to_corpus`
  - [`src/p2_most_common_pair.rs`](src/p2_most_common_pair.rs) — `most_common_pair`
  - [`src/p3_merge_pair.rs`](src/p3_merge_pair.rs) — `merge_pair`
  - [`src/p4_train_bpe.rs`](src/p4_train_bpe.rs) — `train_bpe`
  - [`src/p5a_bpe_encode.rs`](src/p5a_bpe_encode.rs) — `bpe_encode`
  - [`src/p5b_bpe_decode.rs`](src/p5b_bpe_decode.rs) — `bpe_decode`
- **Part II — Trainable Transformer Architecture (Question 6)**:
  - [`src/p6a_linear.rs`](src/p6a_linear.rs) — `Linear` (random $\mathcal{N}(0, 2/\text{in\_dim})$ init)
  - [`src/p6b_embedding.rs`](src/p6b_embedding.rs) — `Embedding` (random $\mathcal{N}(0, 1)$ init)
  - [`src/p6c_silu.rs`](src/p6c_silu.rs) — `silu`
  - [`src/p6d_rms_norm.rs`](src/p6d_rms_norm.rs) — functional `rms_norm` (no learned weight)
  - [`src/p6e_self_attention.rs`](src/p6e_self_attention.rs) — `self_attention`
  - [`src/p6f_multi_head_attention_kv_cache.rs`](src/p6f_multi_head_attention_kv_cache.rs) — `MultiHeadAttentionKVCache`
  - [`src/p6g_mlp.rs`](src/p6g_mlp.rs) — two-layer `MLP`
  - [`src/p6h_transformer_block.rs`](src/p6h_transformer_block.rs) — `TransformerBlock`
  - [`src/p6i_llm.rs`](src/p6i_llm.rs) — `LLM` (trainable `pos_embeddings: Var`, `CausalLM` impl)
- **Part III — Training & Generation**:
  - [`src/p7_cross_entropy_loss.rs`](src/p7_cross_entropy_loss.rs) — arbitrary-rank `cross_entropy_loss`
  - [`src/p8_pretokenize_data.rs`](src/p8_pretokenize_data.rs) — `pretokenize_data`
  - [`src/p9_dataloader.rs`](src/p9_dataloader.rs) — disk-backed `DataLoader`
  - [`src/p10_adam.rs`](src/p10_adam.rs) — `Adam` optimizer keyed by `TensorId`
  - [`src/p11_train_llm.rs`](src/p11_train_llm.rs) — `train_llm`
  - [`src/p12_generate.rs`](src/p12_generate.rs) — `generate_with_writer` + provided `generate`

### Running Tests

Run from `rust/`:

```bash
# Fast, offline local sanity tests (all 22 unit + synthetic end-to-end LLM training tests)
cargo test -p hw5 --test local

# Run a single test while iterating
cargo test -p hw5 --test local test_adam

# Full grader-equivalent submit suite (exact hw5_tests.py inputs + edge cases, offline)
cargo test -p hw5 --test submit
```

### Optional TinyStories Integration Test & Training CLI

To keep `cargo test -p hw5 --test local` and `submit` fast and completely offline, end-to-end
training & evaluation (`eval_llm`) is tested in two tiers:

1. `test_eval_llm_synthetic` (in `tests/local.rs`) and `submit_eval_llm_synthetic` (in
   `tests/submit.rs`) run by default offline in a few seconds, exercising `pretokenize_data`,
   `DataLoader`, `LLM`, `Adam`, `train_llm`, loss drop vs reversed-sequence corruption,
   full-vs-KV-cache equivalence, and `generate_with_writer`.
2. `test_eval_llm_tinystories` (in `tests/local.rs`, marked `#[ignore]`) downloads
   `TinyStoriesV2-GPT4-train.txt` (~2 GB) from HuggingFace, pretokenizes 2 chunks with
   `Gpt2Tokenizer`, trains the Q13 4-layer LLM, and runs the exact `hw5_tests.py` evaluation:

```bash
# Optional: full TinyStories Q13 test (~2 GB download on first run; slow on CPU)
cargo test --release -p hw5 --test local test_eval_llm_tinystories -- --ignored
```

You can also run the interactive BPE demo, small/tiny CPU training, or full Q14 pretraining via
[`examples/train_tinystories.rs`](examples/train_tinystories.rs):

```bash
# BPE tokenizer demo on TinyStories (100K chars -> 2000 tokens, round-trip check)
cargo run --release -p hw5 --example train_tinystories -- --mode bpe

# Fast CPU smoke run (tiny 2-layer model on 64K chars)
cargo run --release -p hw5 --example train_tinystories -- --mode tiny

# Q13 small LLM (2 chunks of 1M chars, 4 layers, dim=256, seq_len=512, batch=16)
cargo run --release -p hw5 --example train_tinystories -- --mode small

# Q14 121M-parameter LLM on full TinyStories
cargo run --release -p hw5 --example train_tinystories -- --mode full

# Custom generation flags
cargo run --release -p hw5 --example train_tinystories -- --mode tiny --prompt "Once upon a time," --temp 0.7 --max-tokens 100
```

> **Local dataset tip**: If you already downloaded `TinyStoriesV2-GPT4-train.txt` (e.g. from the
> Python notebook), set `TINYSTORIES_PATH=/path/to/TinyStoriesV2-GPT4-train.txt` to skip
> downloading from HuggingFace Hub.

---

## PyTorch $\leftrightarrow$ Candle Quick Reference

| PyTorch (`hw5.py`) | Candle (`hw5`) |
|---|---|
| `re.split(r'(?=\s)', text)` | `hw5::split_keep_whitespace(text)` (Rust `regex` lacks lookahead) |
| `[chr(i) for i in range(256)]` | `(0u8..=255).map(\|b\| (char::from(b).to_string(), b as usize))` |
| `Parameter(torch.randn(out, in) * (2/in)**0.5)` | `Var::randn(0.0_f32, (2.0 / in_dim as f64).sqrt() as f32, (out_dim, in_dim), device)?` |
| `Parameter(torch.randn(V, D))` | `Var::randn(0.0_f32, 1.0_f32, (num_tokens, dim), device)?` |
| `np.asarray(tokens, dtype=np.uint16).tobytes()` | `(tok as u16).to_le_bytes()` |
| `np.frombuffer(buf, dtype=np.uint16).astype(np.int64)` | `u16::from_le_bytes([c[0], c[1]]) as u32` $\to$ `Tensor::from_vec(..., DType::U32)` |
| `tiktoken.get_encoding("gpt2")` | `hw5::Gpt2Tokenizer::new()?` (`tiktoken_rs::r50k_base()`, `eot_token = 50256`) |
| `opt.zero_grad(); loss.backward(); opt.step()` | `opt.zero_grad(); let grads = loss.backward()?; opt.step(&grads)?;` |
| `self.u[i], self.v[i]` per `Parameter` | `self.m: HashMap<TensorId, Tensor>`, `self.v: HashMap<TensorId, Tensor>` keyed by `var.id()` |
| `with torch.no_grad(): p -= update` | `p.set(&(p.as_tensor() - update)?)?;` |
| `model(x).float()` | `model.forward(&x, 0, false)?.to_dtype(DType::F32)?` |
| `next(model.parameters()).device` | `model.device()` on `CausalLM` |

### Key Rust/Candle Notes

1. **BPE Tie-Breaking & Insertion Order**:
   Python's `collections.Counter` preserves insertion order, so `text_to_corpus` keeps unique words
   in first-seen order and `most_common_pair` breaks frequency ties in favor of the first-seen pair.
   In Rust, `HashMap` iteration order is randomized, so maintain an explicit `Vec` alongside an
   index `HashMap` to guarantee deterministic first-appearance order.
2. **`Adam` Moment Buffers Keyed by `TensorId`**:
   Every `candle_core::Var` carries a unique `TensorId` (`var.id()`) that stays constant across
   `var.set(&new_tensor)?` updates. Storing `m` and `v` in `HashMap<TensorId, Tensor>` cleanly
   tracks moment state per parameter without relying on slice indices or interior mutability.
3. **Pretokenized Binary Format**:
   `pretokenize_data` writes packed **little-endian `u16`** token IDs (2 bytes per token, matching
   NumPy `uint16` on x86_64/arm64). `DataLoader` seeks to byte offset
   `batch_idx * seq_len * batch_size * 2`, reads `(seq_len + 1) * batch_size * 2` bytes, casts each
   `u16` to `u32` (Candle's integer index dtype), reshapes to `[batch_size, seq_len + 1]`, and
   slices `x = [.., 0..seq_len]` and `y = [.., 1..seq_len + 1]`.

---

## Problem Walkthrough

### Part I — Tokenization (BPE from Scratch)

- **Question 1 — `text_to_corpus`** ([`src/p1_text_to_corpus.rs`](src/p1_text_to_corpus.rs)):
  Split text with `split_keep_whitespace(text)`, deduplicate words while preserving first-appearance
  order, represent each unique word as a `Vec<String>` of single characters, and return `(corpus, counts)`.
- **Question 2 — `most_common_pair`** ([`src/p2_most_common_pair.rs`](src/p2_most_common_pair.rs)):
  Count adjacent token pairs across `corpus` weighted by `counts`, returning the highest-frequency
  pair (breaking ties in favor of the first-seen pair).
- **Question 3 — `merge_pair`** ([`src/p3_merge_pair.rs`](src/p3_merge_pair.rs)):
  Modify `corpus` in place by greedily merging left-to-right occurrences of `pair = (a, b)` into
  `format!("{a}{b}")`.
- **Question 4 — `train_bpe`** ([`src/p4_train_bpe.rs`](src/p4_train_bpe.rs)):
  Initialize the 256 base Latin-1 tokens (`0u8..=255`), then iteratively find `most_common_pair`,
  record the merge and new token ID, and call `merge_pair` until `tokens.len() == num_tokens`.
- **Question 5 — `bpe_encode` & `bpe_decode`** ([`src/p5a_bpe_encode.rs`](src/p5a_bpe_encode.rs), [`src/p5b_bpe_decode.rs`](src/p5b_bpe_decode.rs)):
  Encode by splitting without deduplication, applying `merges` in order, and mapping tokens to IDs;
  decode by inverting `tokens` and concatenating the strings for each ID.

### Part II — Trainable Transformer Architecture (Question 6)

Compared to HW4, note three key changes for training from scratch:
1. **Random initialization**: `Linear` uses $\mathcal{N}(0, 2/\text{in\_dim})$; `Embedding` and
   `LLM::pos_embeddings` use $\mathcal{N}(0, 1)$. All trainable layers expose `parameters(&self) -> Vec<Var>`.
2. **Functional `rms_norm`**: No learned scaling weight — just
   $\text{rms\_norm}(X) = X / \sqrt{\frac{1}{d}\sum_j X_{\dots, j}^2 + \epsilon}$ with $\epsilon = 10^{-5}$.
3. **Two-layer `MLP`**: Replaces `GatedMLP` with $\text{SiLU}(X W_1^T) W_2^T$.

### Part III — Training & Generation

- **Question 7 — `cross_entropy_loss`** ([`src/p7_cross_entropy_loss.rs`](src/p7_cross_entropy_loss.rs)):
  Arbitrary-rank cross-entropy over logits `[..., k]` and integer targets `[...]`, returning a
  rank-0 scalar tensor.
- **Question 8 — `pretokenize_data`** ([`src/p8_pretokenize_data.rs`](src/p8_pretokenize_data.rs)):
  Stream up to `max_chunks` chunks of `chunk_size` characters via `read_char_chunk`, encode each
  chunk with `tokenizer.encode(&chunk)`, and write packed little-endian `u16` token IDs.
- **Question 9 — `DataLoader`** ([`src/p9_dataloader.rs`](src/p9_dataloader.rs)):
  Disk-backed iterator seeking to `batch_idx * seq_len * batch_size * 2`, reading
  `(seq_len + 1) * batch_size` `u16` tokens, and returning `(x, y)` of shape `[batch_size, seq_len]`
  on `device`. Re-iterating `&loader` restarts at batch `0`.
- **Question 10 — `Adam`** ([`src/p10_adam.rs`](src/p10_adam.rs)):
  Bias-corrected Adam optimizer:
  $$
  \begin{aligned}
  m &\leftarrow \beta_1 m + (1 - \beta_1) g, \qquad v \leftarrow \beta_2 v + (1 - \beta_2) g^2 \\
  \hat{m} &\leftarrow \frac{m}{1 - \beta_1^t}, \qquad \hat{v} \leftarrow \frac{v}{1 - \beta_2^t} \\
  w &\leftarrow w - \eta \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
  \end{aligned}
  $$
- **Question 11 — `train_llm`** ([`src/p11_train_llm.rs`](src/p11_train_llm.rs)):
  Iterate over `(x, y)` batches, compute `model.forward(&x, 0, false)?.to_dtype(DType::F32)?`,
  evaluate `cross_entropy_loss`, run `backward()`, and step the optimizer.
- **Question 12 — `generate_with_writer`** ([`src/p12_generate.rs`](src/p12_generate.rs)):
  Autoregressive KV-cached sampling placing token tensors on `model.device()`, streaming decoded
  pieces to `writer`, and stopping when `next_token == tokenizer.eot_token()` or `max_tokens`
  tokens have been sampled.
