# Homework 6 — Supervised Finetuning for Chat & Direct Preference Optimization (DPO)

In this homework, you will finetune a pretrained decoder-only Transformer (using the *exact* same model architecture you built in Homework 5, pretrained for longer on ~1.1 trillion tokens of [FineWeb-EDU](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)) for multi-turn chat using both **Supervised Finetuning (SFT)** and **Direct Preference Optimization (DPO)**.

## Quick Start

From the `rust/` workspace root:

```bash
# Run all local sanity tests (fast, 100% offline)
cargo test -p hw6 --test local

# Run a single problem's local test
cargo test -p hw6 --test local test_dpo_loss

# Run the full submission test suite (fast, 100% offline)
cargo test -p hw6 --test submit

# Optional: run full HuggingFace checkpoint evaluations (~1-4 min in --release)
cargo test --release -p hw6 --test local test_eval_llm_chat -- --ignored
cargo test --release -p hw6 --test local test_eval_llm_dpo -- --ignored
cargo test --release -p hw6 --test submit submit_eval_llm_chat -- --ignored
cargo test --release -p hw6 --test submit submit_eval_llm_dpo -- --ignored
```

## PyTorch $\leftrightarrow$ Candle Design Notes

| Concept | PyTorch (`hw6.py`) | Rust / Candle (`hw6`) |
|---|---|---|
| Checkpoint loading | `torch.load("model_base.pth")` | `LLM::load_weights` loads **both** PyTorch `.pth` (`candle_core::pickle::read_all`) and `.safetensors` (`candle_core::safetensors::load`), casting weights to `DType::F32` on the model's device |
| Checkpoint saving | `torch.save(checkpoint, filename)` | `LLM::save_weights` saves `.safetensors` (`DType::F32` lossless, or `LLM::save_weights_bf16` for compact `BF16` checkpoints) using the exact Llama-like key names (`tok_embeddings.weight`, `pos_embeddings.weight`, `layers.{i}.attention.wo.weight`, ...) |
| Chat tokenizer | `tiktoken.Encoding("gpt2_chat", ...)` | Provided `ChatTokenizer` backed by `tiktoken_rs::CoreBPE` (`r50k_base` + `<USER>=50257`, `</USER>=50258`, `<ASSISTANT>=50259`, `</ASSISTANT>=50300`) |
| Boolean loss masks | `torch.bool` tensor + `logits[mask]` | `DType::U8` tensor (`1` = `true`, `0` = `false`); use provided `mask_indices(&mask)?` with `reshape` + `Tensor::index_select` for masked cross-entropy, or `mask.ne(0u8)?.to_dtype(DType::F32)?` for elementwise multiplication in `log_probs` |
| Frozen DPO reference model | `copy.deepcopy(model)` + `with torch.no_grad():` | **CRITICAL:** Candle's `Var::clone()` is a *shallow* `Arc` clone sharing mutable storage! Always use the provided **`model.deep_clone()?`** to snapshot `model_ref`, and call **`.detach()`** on `model_ref`'s outputs inside `dpo_loss` so `grads.get(ref_param)` stays `None` |
| Stable softplus | `torch.logaddexp(beta * x, torch.zeros_like(x))` | Shifted log-sum-exp with detached max $m = \max(\beta x, 0).\text{detach}()$: $m + \log(e^{-m} + e^{\beta x - m})$ — overflow-free and gives exact gradient $\frac{\beta}{2}$ at $x = 0$ |

## Provided Scaffolding (Do Not Modify)

- `src/nn.rs` — `CausalLM`, `Optimizer`, `multinomial`, and `mask_indices`.
- `src/chat.rs` — `Message`, `Tokenizer`, `Gpt2Tokenizer`, `ChatTokenizer`, `ModelConfig` (`params.json`), `ChatFiles`, `download_hw6_files`, `load_checkpoint_tensors`, and `load_pretrained_llm`.
- Inside `src/p0k_llm.rs` — `LLM::load_weights`, `LLM::save_weights`, `LLM::save_weights_bf16`, `LLM::deep_clone`, and `impl CausalLM for LLM`.
- Inside `src/p0l_generate.rs` — `generate` wrapper around `generate_with_writer`.

---

## Part 0 — Setting Up the Model

Copy your working HW5 implementations into `src/p0a_linear.rs` through `src/p0l_generate.rs`:

- `src/p0a_linear.rs` — `Linear`
- `src/p0b_embedding.rs` — `Embedding`
- `src/p0c_silu.rs` — `silu`
- `src/p0d_rms_norm.rs` — `rms_norm`
- `src/p0e_self_attention.rs` — `self_attention`
- `src/p0f_multi_head_attention_kv_cache.rs` — `MultiHeadAttentionKVCache`
- `src/p0g_mlp.rs` — `MLP`
- `src/p0h_transformer_block.rs` — `TransformerBlock`
- `src/p0i_cross_entropy_loss.rs` — `cross_entropy_loss`
- `src/p0j_adam.rs` — `Adam`
- `src/p0k_llm.rs` — `LLM` (`new`, `forward`, `parameters`; keep provided `load_weights`, `save_weights`, `deep_clone`)
- `src/p0l_generate.rs` — `generate_with_writer` (**updated** from HW5 to take `eot_token: Option<u32>` so chat generation can stop on `</ASSISTANT>` = `Some(50300)`)

---

## Part I — Chat Training via Supervised Finetuning (SFT)

We finetune the pretrained FineWeb-EDU base model on a filtered subset of [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) (`ultrachat_short.json`) containing conversations of at most 1024 tokens.

### Question 1 — Conversion to Chat Format (`src/p1_convert_to_chat_format.rs`)

Implement `convert_to_chat_format(messages: &[Message]) -> String` converting UltraChat messages into a single concatenated string using `<USER>`, `</USER>`, `<ASSISTANT>`, and `</ASSISTANT>` tags with no extra separators or whitespace:

```text
<USER>This is a user question.</USER><ASSISTANT>This is an assistant response.</ASSISTANT>
```

### Question 2 — Pretokenizing Chat Data (`src/p2_pretokenize_chat.rs`)

Implement `pretokenize_chat(tokenizer, in_filename, out_filename)` to read a JSON array of conversations (`Vec<Vec<Message>>`), format each with `convert_to_chat_format`, encode each with `tokenizer.encode(&text)` (which allows all special tokens), and write the resulting `Vec<Vec<u32>>` JSON to `out_filename`.

Note that `ChatTokenizer` adds the exact special tokens from `hw6.py` onto GPT-2 (`r50k_base`):
- `<|endoftext|>` $\to$ `50256`
- `<USER>` $\to$ `50257`
- `</USER>` $\to$ `50258`
- `<ASSISTANT>` $\to$ `50259`
- `</ASSISTANT>` $\to$ `50300`

### Question 3 — Chat Masking (`src/p3_get_loss_mask.rs`)

Implement `get_loss_mask(tokens: &[u32], tokenizer: &T) -> Vec<bool>` returning `true` **only** for tokens occurring strictly *after* `<ASSISTANT>` up through (and *including*) `</ASSISTANT>`.

### Question 4 — Data Loader for Chat (`src/p4_dataloader_chat.rs`)

Implement `DataLoaderChat` backed by a pretokenized JSON file (`Vec<Vec<u32>>`):
- Each row contains a **single** conversation (needed for paired DPO batches).
- `x` (`[batch_size, seq_len]`, `U32`): `chat[0..seq_len]` zero-padded with `0`.
- `y` (`[batch_size, seq_len]`, `U32`): shifted next-token targets `chat[1..seq_len + 1]` zero-padded with `0`.
- `mask` (`[batch_size, seq_len]`, `U8`): `get_loss_mask(chat)[1..seq_len + 1]` padded with `0` (`false`).
- Drop incomplete trailing batches (`len < batch_size`).

### Question 5 — Chat Training Loop (`src/p5_train_llm_chat.rs`)

Implement `train_llm_chat(model, chat_loader, opt, max_iter)` to run masked next-token SFT using `mask_indices(&mask)?` + `index_select` and `cross_entropy_loss`, stopping after `max_iter` steps when `max_iter` is `Some(limit)`.

---

## Part II — Direct Preference Optimization (DPO)

Starting from the SFT chat model (`model_chat.pth`), we further align the model using human/preference pairs (`ultrachat_dpo_pos.json` vs `ultrachat_dpo_neg.json`).

### Question 6 — Log Probability Calculation (`src/p6_log_probs.rs`)

Implement `log_probs(logits, y, mask)` returning a `[batch_size]` tensor containing the **sum** of masked per-token log probabilities for each batch element:

$$\log p(y_b \mid x_b) = \sum_{t=0}^{S-1} \mathbb{1}[\text{mask}_{b,t} \ne 0] \cdot \left( z_{b, t, y_{b,t}} - \log \sum_{j=0}^{V-1} e^{z_{b, t, j}} \right)$$

### Question 7 — Softplus & DPO Loss (`src/p7a_softplus.rs`, `src/p7b_dpo_loss.rs`)

Given a prompt $x$, preferred completion $y^+$, and dispreferred completion $y^-$, the DPO loss is:

$$L_{\text{DPO}} = \text{softplus}\!\left(-\log\frac{p(y^+ \mid x)}{p_{\text{ref}}(y^+ \mid x)} + \log\frac{p(y^- \mid x)}{p_{\text{ref}}(y^- \mid x)},\; \beta\right)$$

where

$$\text{softplus}(x, \beta) = \log\!\left(1 + e^{\beta x}\right) = \text{logaddexp}(0, \beta x).$$

1. Implement numerically stable `softplus(x, beta)` in `src/p7a_softplus.rs` without overflowing for large positive inputs and preserving the exact derivative $\frac{\beta}{2}$ at $x = 0$.
2. Implement `dpo_loss(model, model_ref, xp, yp, maskp, xn, yn, maskn, beta)` in `src/p7b_dpo_loss.rs`, ensuring `model_ref` outputs are `.detach()`ed so no gradients flow into `model_ref`.

### Question 8 — DPO Training Loop (`src/p8_train_dpo.rs`)

Implement `train_dpo(model, model_ref, loader_pos, loader_neg, opt, beta, max_iter)` iterating over zipped `(pos_batch, neg_batch)` pairs, minimizing `dpo_loss(...).mean_all()?`, and stopping after `max_iter` steps if provided.

---

## CLI Runner (`examples/train_chat_dpo.rs`)

Replaces the notebook's interactive download, SFT, DPO, and generation cells:

```bash
# Generate from the pretrained FineWeb-EDU base model (`model_base.pth`)
cargo run --release -p hw6 --example train_chat_dpo -- --mode base

# Pretokenize UltraChat SFT + DPO JSON files (use `--limit 256` for a quick subset)
cargo run --release -p hw6 --example train_chat_dpo -- --mode pretokenize --limit 256

# Run 10 CPU steps of SFT chat finetuning & save checkpoint to `.safetensors`
cargo run --release -p hw6 --example train_chat_dpo -- --mode sft --max-iter 10 --save /tmp/hw6_sft.safetensors

# Run 10 CPU steps of DPO finetuning starting from `model_chat.pth`
cargo run --release -p hw6 --example train_chat_dpo -- --mode dpo --max-iter 10 --beta 0.1 --save /tmp/hw6_dpo.safetensors

# Interactive chat with `model_chat.pth` (or pass `--checkpoint /tmp/hw6_dpo.safetensors`)
cargo run --release -p hw6 --example train_chat_dpo -- --mode chat
```

Set `HW6_DIR=/path/to/files` if you have already downloaded the 6 HuggingFace files locally.
