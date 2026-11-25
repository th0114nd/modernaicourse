# Intro to Modern AI — Rust + Candle Port

This workspace is a Rust/[Candle](https://github.com/huggingface/candle) port of CMU's
[Intro to Modern AI](https://modernaicourse.org) (`modernaicourse`) homeworks.

The original Python/PyTorch marimo notebooks remain in [`../homework/`](../homework/) for
side-by-side reference.

## Structure

Each homework is a crate in this Cargo workspace:

- `hwN/README.md` — Course prose, mathematical background, and problem instructions (ported from the marimo notebook).
- `hwN/src/p*_*.rs` — One module per problem with `todo!()` stubs between `// BEGIN YOUR CODE` and `// END YOUR CODE`.
- `hwN/tests/local.rs` — Local sanity-check unit tests (ported from `test_*` in `hwN_tests.py`).
- `hwN/tests/submit.rs` — Grader-equivalent tests (ported from `submit_*` in `hwN_tests.py` with verified ground-truth assertions plus extra edge cases).

## Workflow

Work through each homework crate in order:

```bash
cd rust

# Run local sanity tests while iterating
cargo test -p hw0 --test local

# Run a single problem's local test
cargo test -p hw0 --test local test_primes

# When ready to check against the full grader suite ("submit")
cargo test -p hw0 --test submit
```

Tip: for a marimo-like live feedback loop on save, use `cargo watch`:

```bash
cargo watch -x "test -p hw0 --test local"
```

## Homeworks

- [hw0](hw0/README.md) — Workspace setup and Rust programming basics
- [hw1](hw1/README.md) — Introduction to linear algebra and Candle tensors
- [hw2](hw2/README.md) — Automatic differentiation from scratch and MNIST linear classifier
- [hw3](hw3/README.md) — Training neural networks (`Linear`, `SGD`, `DataLoader`, `TwoLayerNN`, `MultiLayerNN`) on MNIST
- [hw4](hw4/README.md) — Transformers & `Llama3Simplified` (`Linear`, `Embedding`, `silu`, `RMSNorm`, `self_attention`, `MultiHeadAttention`, KV cache, `GatedMLP`, `TransformerBlock`, `generate`)
- [hw5](hw5/README.md) — BPE tokenizer from scratch (`train_bpe`, `bpe_encode`/`bpe_decode`) & pretraining an LLM (`MLP`, `TransformerBlock`, `LLM`, `cross_entropy_loss`, `pretokenize_data`, disk-backed `DataLoader`, `Adam`, `train_llm`, `generate`) on TinyStories
- [hw6](hw6/README.md) — Supervised finetuning for chat & Direct Preference Optimization (`convert_to_chat_format`, `pretokenize_chat`, `get_loss_mask`, `DataLoaderChat`, `train_llm_chat`, `log_probs`, `softplus`, `dpo_loss`, `train_dpo`, `.pth`/`.safetensors` weight I/O & `LLM::deep_clone`) on UltraChat
- [hw7](hw7/README.md) — Reasoning models & Reinforcement Learning on GSM8K (batched KV-cache `MultiHeadAttentionKVCache`, `generate_parallel`, `gsm8k_to_text`, `pretokenize_gsm8k`, `get_loss_mask`, dynamically-padded `DataLoader`, `train_llm_sft`, `eval_tool`, `generate_parallel_tool`, `extract_answer`, `grade_responses`, `eval_model`, `rl_loss`, `train_llm_rl`)
