# Modern AI Course — Rust/Candle Port Handoff & Plan

This document records the agreed architecture, conventions, verification workflow, and
homework-by-homework roadmap for porting CMU's [Intro to Modern AI](https://modernaicourse.org)
from Python/PyTorch (marimo notebooks) to Rust/Candle.

## 1. Repository & Worktree Context

- **Worktree path**: `/home/thnd/modernaicourse-rust` (branch `rust`, branched from `main` `64f7811`)
- **Original repo**: `/home/thnd/modernaicourse` (user's Python work — **never modify**)
- **Rust workspace**: `/home/thnd/modernaicourse-rust/rust/`
- **Reference notebooks**: `/home/thnd/modernaicourse-rust/homework/hw*/hw*.py` and `hw*_tests.py` remain untouched as side-by-side reference.

> **Important**: The user is actively solving completed homeworks in this worktree (e.g. `hw0/src/*.rs`).
> **Never overwrite or reset `src/` files of already-completed homeworks** unless explicitly asked!

---

## 2. Agreed Design Decisions

1. **ML Framework**: **Candle** (`candle-core`, `candle-nn`)
   - CPU baseline; add optional `cuda` feature flag where relevant.
   - Dynamic-rank `candle_core::Tensor` matches PyTorch's ergonomics across arbitrary-rank ops (`batch_matmul`, `Linear`, `RMSNorm`, attention) without Burn's const-generic rank friction.
   - Native PyTorch `.pth` checkpoint loading via `candle_core::pickle::read_all` (needed for HW4, HW6, HW7).
   - Default floating-point dtype: `DType::F32` (matching PyTorch defaults).

2. **Crate & Module Layout**:
   - Cargo workspace under `rust/` with:
     - `common/` — shared utilities (MNIST IDX loader with caching, `assert_allclose` tensor assertion helpers, shared test helpers).
     - `hw0/`, `hw1/`, ..., `hw7/` — one crate per homework.
   - Inside each `hwN/`:
     - `README.md` — full course prose, LaTeX math, and exercise walkthrough ported from `homework/hwN/hwN.py`.
     - `src/lib.rs` — module index and public re-exports.
     - `src/p1_*.rs`, `src/p2_*.rs`, ... — **one module file per problem** with docstrings, exact signatures, `#[allow(unused_variables)]`, and `todo!()` inside `// BEGIN YOUR CODE` / `// END YOUR CODE`.
     - Any provided starter types/helpers (e.g. `hw0::Polynomial`, `hw1::add_matmul_44`) live in their own clearly labeled module ("do not modify").
     - `examples/` — runnable binaries replacing marimo's interactive "Run training / Chat" buttons (starting in HW2/HW3+).

3. **Two-Stage Test Suites (`local` & `submit`)**:
   - Tests live in **`tests/local.rs`** and **`tests/submit.rs`** (separate from `src/` to limit temptation).
   - **CRITICAL Neovim/Neotest convention**: Wrap all tests in `mod local { ... }` inside `tests/local.rs` and `mod submit { ... }` inside `tests/submit.rs`. `rustaceanvim.neotest` requires a `mod` namespace to expose a file-level runnable!
   - `tests/local.rs`: direct port of `test_*` sanity checks from `hwN_tests.py`.
   - `tests/submit.rs`: port the exact deterministic inputs from `submit_*` in `hwN_tests.py` with verified ground-truth assertions (computed analytically, via reference implementations, or via Candle native ops) PLUS thoughtfully designed extra edge cases.
   - **Honor system**: Do *not* replicate Python's AST/bytecode/dispatch anti-cheat hacks (`PreventTorchOps`, `dis.get_instructions`); document constraints clearly in `README.md` and doc comments.

4. **Verification Protocol (Required for Every Homework)**:
   1. Implement full working reference solutions in `src/p*.rs`.
   2. Run and verify **100% pass** on:
      - `cargo test -p hwN --test local`
      - `cargo test -p hwN --test submit`
      - `cargo clippy -p hwN --all-targets -- -D warnings`
   3. Replace `src/p*.rs` bodies back with clean `todo!()` stubs (keeping `#[allow(unused_variables)]` so stubs compile warning-free).
   4. Verify `cargo test -p hwN --no-run` and `cargo clippy -p hwN --all-targets -- -D warnings` succeed cleanly on the stubs.
   5. Update `rust/README.md` and the status table below.

---

## 3. Homework-by-Homework Roadmap

| HW | Topic | Status | Key Dependencies & Porting Notes |
|----|-------|--------|----------------------------------|
| `common` | Shared MNIST loader + tensor test helpers | **Done** | Pure-Rust IDX3/IDX1 parser + `ureq`/`flate2` download cache (`~/.cache/modernaicourse/mnist`, also checks `../../modernaicourse/homework/hw1/MNIST/raw` if present); `assert_allclose`, `linspace`, `seeded_randn` |
| `hw0` | Setup & Rust basics (`add`, `primes`, `Polynomial`) | **Done** | Pure std Rust; 5/5 local + 5/5 submit verified; user actively solving |
| `hw1` | Linear algebra & Candle tensors (`classify_zero_one`, vector/matrix/batch/block matmul) | **Done** | `candle-core`, `common`; 11 problem modules + provided `add_matmul_44`; 11/11 local + 11/11 submit verified against reference solutions, restored to clean stubs |
| `hw2` | Autograd from scratch (`Variable`/`Function`/`compute_gradients`) + MNIST linear model (`cross_entropy_loss`, `error`, `train_sgd`) | **Done** | `candle-core`, `common`; provided `autograd.rs` (`Rc<RefCell<...>>` `Variable`, `Function` trait, `Multiply`, `Negate`), 10 problem modules (`p1a`–`p1f`, `p2`, `p3a`–`p3b`, `p4`), and `examples/train_mnist.rs`; 10/10 local + 11/11 submit verified against reference solutions (incl. `hw2_tests.py` exact inputs + MNIST training run reaching 8.32% test error in 5 epochs), restored to clean stubs |
| `hw3` | Neural networks (`Linear`, `CrossEntropyLoss`, `SGD`, `DataLoader`, `epoch`, `TwoLayerNN`, `MultiLayerNN`) on MNIST | **Done** | `candle-core`, `common`; provided `nn.rs` (`Module`, `Loss`, `Optimizer` traits + `load_mnist_dataloaders`, `train_model`), 9 problem modules (`p1`–`p9`), and `examples/train_mnist.rs`; 9/9 local + 9/9 submit verified against reference solutions (`--release` ~28s each; linear 8.32% err in 2.5s, 2-layer NN 2.17% err in 29s), restored to clean stubs |
| `hw4` | Transformer & `Llama3Simplified` (`Linear`, `Embedding`, `silu`, `RMSNorm`, `self_attention`, `MultiHeadAttention`, `MultiHeadAttentionKVCache`, `GatedMLP`, `TransformerBlock`, `Llama3Simplified`, `generate`) | **Done** | `candle-core`, `hf-hub`, `tiktoken-rs`, `serde`/`serde_json`, `rand`; provided `llama.rs` (`CausalLM`, `LlamaConfig`, `download_llama_files`, `load_pth_checkpoint` via `pickle::read_all`, `load_pretrained_llama3`, `multinomial`) and `tokenizer.rs` (`Tokenizer`, `LlamaTokenizer`, `Message`, `ChatFormat`), 11 problem modules (`p1`–`p10`), and `examples/chat.rs`; `&mut self` KV-cache forward; 12/12 local (+ 1 `#[ignore]` 1B pretrained test) & 11/11 submit verified against reference solutions, restored to clean stubs |
| `hw5` | BPE tokenizer from scratch + LLM pretraining (`MLP`, `TransformerBlock`, `LLM`, `Adam`, `DataLoader`, `train_llm`, `generate`, TinyStories) | **Done** | `candle-core`, `hf-hub`, `tiktoken-rs`, `rand`; provided `nn.rs` (`CausalLM`, `Optimizer`, `multinomial`, `split_keep_whitespace`, `read_char_chunk`) and `tinystories.rs` (`Tokenizer`, `Gpt2Tokenizer` via `r50k_base`, `padded_vocab_size`, `download_tinystories`), 21 problem modules (`p1`–`p5b`, `p6a`–`p6i`, `p7`–`p12`), and `examples/train_tinystories.rs`; `Adam` moment buffers keyed by `TensorId`; packed little-endian `u16` pretokenized binary format; 22/22 local (+ 1 `#[ignore]` TinyStories test) & 22/22 submit verified against reference solutions, restored to clean stubs |
| `hw6` | SFT for Chat & DPO (reuse LLM + `Adam` + configurable-EOT `generate`, `load_weights`/`save_weights`/`deep_clone`, `convert_to_chat_format`, `pretokenize_chat`, `get_loss_mask`, `DataLoaderChat`, `train_llm_chat`, `log_probs`, `softplus`, `dpo_loss`, `train_dpo`) | **Done** | `candle-core`, `hf-hub`, `tiktoken-rs`, `rustc-hash`, `serde`/`serde_json`, `rand`; provided `nn.rs` (`CausalLM`, `Optimizer`, `multinomial`, `mask_indices`) and `chat.rs` (`Message`, `Tokenizer`, `Gpt2Tokenizer`, `ChatTokenizer` with `<USER>=50257`, `</USER>=50258`, `<ASSISTANT>=50259`, `</ASSISTANT>=50300`, `ModelConfig`, `download_hw6_files`, `load_checkpoint_tensors`, `load_pretrained_llm`), 21 problem modules (`p0a`–`p0l`, `p1`–`p5`, `p6`, `p7a`–`p7b`, `p8`), and `examples/train_chat_dpo.rs`; `LLM::load_weights` supports both `.pth` (`pickle::read_all`) and `.safetensors`, `LLM::save_weights`/`save_weights_bf16` write `.safetensors`, and `LLM::deep_clone` allocates independent `Var`s for frozen DPO `model_ref`; 21/21 local (+ 2 `#[ignore]` HF eval tests) & 11/11 submit (+ 2 `#[ignore]` HF eval tests) verified against reference solutions, restored to clean stubs |
| `hw7` | Reasoning models & RL on GSM8K (batched `MultiHeadAttentionKVCache`, `generate_parallel`, `gsm8k_to_text`, `pretokenize_gsm8k`, `get_loss_mask`, dynamically-padded `DataLoader`, `train_llm_sft`, `eval_tool`, `generate_parallel_tool`, `extract_answer`, `grade_responses`, `eval_model`, `rl_loss`, `train_llm_rl`) | **Done** | `candle-core`, `hf-hub`, `tiktoken-rs`, `regex`, `rustc-hash`, `serde`/`serde_json`, `rand`, `ureq`; provided `nn.rs` (`CausalLM`, `Optimizer`, `multinomial`, 2D `multinomial_batch`, `mask_indices`) and `gsm8k.rs` (`Gsm8kExample`, `Tokenizer`, `Gpt2Tokenizer`, `ReasoningTokenizer` with `<QUESTION>`..`</ANSWER>` mapped to `50257..=50266`, safe recursive-descent `eval_arithmetic_expr`, `ModelConfig`, `download_hw7_files` via `hf-hub` `zkolter/RL-Homework` + GSM8K JSONL $\to$ JSON cache with `HW7_DIR` override, `load_checkpoint_tensors`, `load_pretrained_llm`), 25 problem modules (`p0a`–`p0k`, `p1`–`p13` incl. `p10a`/`p10b`), and `examples/train_reasoning.rs`; 25/25 local (+ 1 `#[ignore]` HF SFT eval test) & 14/14 submit (+ 1 `#[ignore]` HF SFT eval test) verified against reference solutions, restored to clean stubs |

---

## 4. Technical Conventions to Maintain Across Crates

- **Error type**: Use `candle_core::Result<T>` everywhere Candle tensors are involved.
- **Immutability**: Candle `Tensor` is immutable; trainable parameters use `candle_core::Var`.
  - Where PyTorch mutates weights under `torch.no_grad()`, Candle uses `var.set(&new_tensor)?`.
  - Where PyTorch uses `loss.backward(); optimizer.step(); optimizer.zero_grad()`, Candle uses `let grads = loss.backward()?; optimizer.step(&grads)?;` (keep a no-op `zero_grad()` or document why `GradStore` makes `zero_grad` unnecessary).
- **Randomness / determinism in tests**: Seed tests deterministically (e.g. `Device::Cpu` + deterministic `linspace` or seeded `rand` buffers via `Tensor::from_vec`) so tests never flake.
- **Float tolerance**: Use `assert_allclose(actual, expected, 1e-4, 1e-4)` (or tighter `1e-5` where appropriate) for `f32` tensor comparisons.
- **Keep `cargo clippy --all-targets -- -D warnings` clean** for both verified reference solutions AND final stubbed crates.
