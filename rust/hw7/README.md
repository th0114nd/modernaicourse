# Homework 7 — Reasoning Models & Reinforcement Learning on GSM8K

In this final homework, you will use **supervised finetuning (SFT)**, **live tool execution**, and **reinforcement learning (RL)** to build a minimal reasoning model that can solve grade-school math problems from the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) dataset by writing step-by-step reasoning traces and invoking an arithmetic calculator tool.

For example, given a GSM8K problem:
```text
Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?
```
your model will generate tagged reasoning traces (`<THINK>...</THINK>`), call `<TOOL>...</TOOL>` for arithmetic operations, receive live `<RESPONSE>...</RESPONSE>` feedback injected mid-generation, and emit a final integer answer inside `<ANSWER>...</ANSWER>`.

## Crate Layout & Workflow

- **Provided scaffolding (do not modify)**:
  - `src/nn.rs` — `CausalLM`, `Optimizer`, `multinomial`, batched 2D `multinomial_batch` (Candle counterpart to `torch.multinomial(probs, 1)`), and `mask_indices`.
  - `src/gsm8k.rs` — `Gsm8kExample`, `Tokenizer`, `Gpt2Tokenizer`, `ReasoningTokenizer` (GPT-2 `r50k_base` augmented with the exact 10 HW7 special tokens `50257..=50266`), safe arithmetic evaluator `eval_arithmetic_expr` (Rust counterpart to Python's `eval()`), `ModelConfig`, `download_hw7_files` (`zkolter/RL-Homework` + GSM8K JSONL $\to$ JSON caching with `HW7_DIR` override), and `.pth`/`.safetensors` weight loaders.
- **Part 0 — Carried-Over LLM & Training Components (`src/p0a_*.rs`–`src/p0k_*.rs`, local tests only)**:
  Copy your working implementations from `hw6`:
  - `p0a_linear.rs` (`Linear`), `p0b_embedding.rs` (`Embedding`), `p0c_silu.rs` (`silu`), `p0d_rms_norm.rs` (`rms_norm`), `p0e_self_attention.rs` (`self_attention`), `p0f_mlp.rs` (`MLP`), `p0g_transformer_block.rs` (`TransformerBlock`), `p0h_cross_entropy_loss.rs` (`cross_entropy_loss`), `p0i_adam.rs` (`Adam`), `p0j_llm.rs` (`LLM` — keep provided `load_weights`, `save_weights`, `save_weights_bf16`, `deep_clone`, `CausalLM`), `p0k_log_probs.rs` (`log_probs`).
- **Parts I–IV — Graded Problems (`src/p1_*.rs`–`src/p13_*.rs`, both `local` and `submit` tests)**:
  1. `p1_multi_head_attention_kv_cache.rs` — Multi-batch KV-cached `MultiHeadAttentionKVCache`
  2. `p2_generate_parallel.rs` — Parallel autoregressive sampling (`generate_parallel`)
  3. `p3_gsm8k_to_text.rs` — Convert GSM8K `{question, answer}` to tagged reasoning text (`gsm8k_to_text`)
  4. `p4_pretokenize_gsm8k.rs` — Pretokenize GSM8K JSON files (`pretokenize_gsm8k`)
  5. `p5_get_loss_mask.rs` — Reasoning & answer loss mask (`get_loss_mask`)
  6. `p6_dataloader.rs` — Dynamically-padded GSM8K `DataLoader`
  7. `p7_train_llm_sft.rs` — Masked cross-entropy SFT loop (`train_llm_sft`)
  8. `p8_eval_tool.rs` — Arithmetic tool evaluator with near-integer rounding (`eval_tool`)
  9. `p9_generate_parallel_tool.rs` — Parallel generation with live tool execution (`generate_parallel_tool`)
  10. `p10a_extract_answer.rs` (`extract_answer`) & `p10b_grade_responses.rs` (`grade_responses`)
  11. `p11_eval_model.rs` — GSM8K Pass@1, formatting rate, and Pass@k evaluation (`eval_model`)
  12. `p12_rl_loss.rs` — Mean-centered policy-gradient loss (`rl_loss`)
  13. `p13_train_llm_rl.rs` — RL training loop (`train_llm_rl`)

### Running Tests

Run from `rust/`:
```bash
# Run all local sanity tests (fast, 100% offline)
cargo test -p hw7 --test local

# Run a single problem's local test
cargo test -p hw7 --test local test_rl_loss

# Run all grader-equivalent submission tests (fast, 100% offline)
cargo test -p hw7 --test submit

# Optional: download model_sft.pth + GSM8K and run pretrained evaluation smoke tests
cargo test --release -p hw7 --test local test_eval_sft_pretrained -- --ignored
cargo test --release -p hw7 --test submit submit_eval_sft_pretrained -- --ignored
```

---

## Part I — Parallel Sampling

RL requires generating many completions per prompt to estimate reward baselines and policy gradients. To sample `num_completions` trajectories in one forward pass, we generalize the KV cache from batch size `1` to `max_cache_batches`.

### Question 1 — Multi-batch KV Cache (`src/p1_multi_head_attention_kv_cache.rs`)

Allocate `k_cache` and `v_cache` with shape `[max_cache_batches, max_cache_size, dim]` (default `DEFAULT_MAX_CACHE_BATCHES = 64`). When `use_kv_cache` is `true` for an input `x` of shape `[B, S, dim]`:
- Check `B <= max_cache_batches` and `seq_pos + S <= max_cache_size`.
- Write detached keys/values into slice `[0..B, seq_pos..seq_pos + S, 0..dim]` using `Tensor::slice_assign`.
- Read back `k_cache.narrow(0, 0, B)?.narrow(1, 0, seq_pos + S)?` (and similarly for `v_cache`).

> **Candle note**: Always call `.detach()` before writing `k` and `v` into `self.k_cache` / `self.v_cache` so the cache never retains stale autograd graphs across steps.

### Question 2 — Parallel Generation (`src/p2_generate_parallel.rs`)

Implement `generate_parallel` to sample `num_completions` completions from a single `prompt_tokens` slice:
- `max_tokens` is the **total** sequence length (prompt + generated tokens).
- Pre-allocate a `[num_completions, max_tokens]` token buffer (zero-initialized).
- Use `multinomial_batch(&probs)` (Candle counterpart to `torch.multinomial(probs, 1)`) to sample one token per row simultaneously.
- When `eot_token` is `Some(id)`, stop early **only after every completion** has generated `id`. Completions that hit `id` early continue generating normally until all rows finish.

---

## Part II — Supervised Finetuning on GSM8K

Each GSM8K example has `"question"` and `"answer"` fields, where `"answer"` embeds calculator annotations `<<expr=val>>` and terminates with `\n#### <integer>`.

### Question 3 — Converting GSM8K to Tagged Text (`src/p3_gsm8k_to_text.rs`)

Convert an example into:
```text
<QUESTION>{question}</QUESTION><THINK>{reasoning}</THINK><ANSWER>{final_answer}</ANSWER>
```
replacing every `<<expr=val>>` inside `{reasoning}` with `<TOOL>{expr}</TOOL><RESPONSE>{val}</RESPONSE>`.

### Question 4 — Pretokenizing GSM8K (`src/p4_pretokenize_gsm8k.rs`)

Read a JSON array of `Gsm8kExample`, format each with `gsm8k_to_text`, encode using `tokenizer.encode(&text)`, and write `Vec<Vec<u32>>` as JSON.

The `ReasoningTokenizer` in `src/gsm8k.rs` extends GPT-2 (`r50k_base`) with the exact contiguous special token IDs from `hw7.py`:

| Token | ID | Token | ID |
|---|---:|---|---:|
| `<QUESTION>` | `50257` | `</QUESTION>` | `50258` |
| `<THINK>` | `50259` | `</THINK>` | `50260` |
| `<TOOL>` | `50261` | `</TOOL>` | `50262` |
| `<RESPONSE>` | `50263` | `</RESPONSE>` | `50264` |
| `<ANSWER>` | `50265` | `</ANSWER>` | `50266` |

### Question 5 — Loss Masking (`src/p5_get_loss_mask.rs`)

Build a boolean mask selecting only model-generated reasoning and answer tokens:
- Start `false` (ignore `<QUESTION>...</QUESTION><THINK>`).
- Turn `true` **after** `<THINK>` and **after** `</RESPONSE>`.
- Turn `false` **after** `</TOOL>` (don't train to predict calculator output) and **after** `</ANSWER>` (ignore post-answer padding).

### Question 6 — Dynamically-Padded GSM8K `DataLoader` (`src/p6_dataloader.rs`)

Load pretokenized `Vec<Vec<u32>>` JSON, precompute masks with `get_loss_mask`, drop any incomplete trailing batch, and for each batch dynamically pad every row to `seq_len = max_len - 1` (where `max_len` is the longest sequence in that batch):
- `x`: `[batch_size, seq_len]` (`DType::U32`), zero-padded
- `y`: `[batch_size, seq_len]` (`DType::U32`), shifted left by 1, zero-padded
- `mask`: `[batch_size, seq_len]` (`DType::U8`), shifted left by 1, padded with `0`

### Question 7 — Supervised Finetuning (`src/p7_train_llm_sft.rs`)

Identical to HW6's `train_llm_chat`: extract masked logits and targets with `mask_indices`, compute `cross_entropy_loss`, backpropagate, and step the optimizer (respecting optional `max_iter`).

---

## Part III — Evaluating Tools & Reasoning Models

### Question 8 — Tool Evaluation (`src/p8_eval_tool.rs`)

Evaluate arithmetic expressions from `<TOOL>...</TOOL>` using the provided `eval_arithmetic_expr` (Rust's safe counterpart to Python's `eval()`):
- On error (syntax error, division by zero, non-finite result), return `"ERROR"`.
- If the result $v$ is within $10^{-4}$ of an integer ($|v - \text{round}(v)| \le 10^{-4}$), return the integer string (e.g., `"2"`, `"24"`).
- Otherwise return the floating-point string (e.g., `"3.5"`, `"1.75"`).

### Question 9 — Generation with Tool Calls (`src/p9_generate_parallel_tool.rs`)

Extend `generate_parallel` per batch element independently:
1. When completion $i$ emits `</TOOL>`, locate its most recent `<TOOL>`, decode the tokens strictly between them, call `eval_tool` (or `"ERROR"` if no `<TOOL>` exists), tokenize `<RESPONSE>{res}</RESPONSE>`, and enqueue those tokens to overwrite completion $i$'s next $n$ sampled tokens (still passing them through the model so the KV cache stays synchronized!).
2. Once completion $i$ emits `</ANSWER>`, force all subsequent tokens for completion $i$ to `0`.
3. Stop early once every completion has emitted `eot_token` (when `Some`).

### Question 10 — Extracting & Grading Answers (`src/p10a_extract_answer.rs`, `src/p10b_grade_responses.rs`)

- `extract_answer`: decode the tokens strictly between `<ANSWER>` and `</ANSWER>` and parse as `i64`, returning `None` on failure.
- `grade_responses`: score each completion row as `format_weight` (if `extract_answer` returns `Some`) plus `correct_weight` (if it equals the ground-truth answer).

### Question 11 — Evaluating the Reasoning Model (`src/p11_eval_model.rs`)

Iterate up to `max_cases` examples from a batch-size-1 loader. For each example, slice `x[0]` up to and including `<THINK>` as the prompt, sample `num_completions` completions via `generate_parallel_tool` (with `eot_token = Some(</ANSWER>)`), grade against `y[0]`, and return `(pass@1, format_rate, pass@k)`.

---

## Part IV — Reinforcement Learning

### Question 12 — RL Loss (`src/p12_rl_loss.rs`)

Given $N$ sampled completions $y_1, \ldots, y_N$ for a prompt $x$ with scalar rewards $R(x, y_i)$ and batch mean $\bar{R} = \frac{1}{N}\sum_{j=1}^{N} R(x, y_j)$, compute the mean-centered policy-gradient loss:

$$\mathcal{L}_{\text{RL}} = -\frac{1}{N_{\text{tok}}} \sum_{i=1}^{N} \log p(y_i \mid x) \cdot \left(R(x, y_i) - \bar{R}\right)$$

where $\log p(y_i \mid x)$ is computed by `log_probs` over `(logits, tokens[:, 1:], mask[:, 1:])` using `get_loss_mask`, and $N_{\text{tok}}$ is the total number of `true` tokens in the mask across the batch.

> **PyTorch $\leftrightarrow$ Candle Notes**:
> - Note the leading **minus sign** so minimizing $\mathcal{L}_{\text{RL}}$ via gradient descent maximizes expected reward (matching `hw7_tests.py`).
> - If all rewards in a group are identical (zero variance), $R(x, y_i) - \bar{R} = 0$ cleanly produces a `0.0` loss with no division by zero.
> - If you wish to experiment with a KL-divergence penalty against a frozen SFT reference policy, always create the reference copy with `model.deep_clone()?` (never `.clone()`, which shares `Var` storage in Candle).

### Question 13 — RL Training Loop (`src/p13_train_llm_rl.rs`)

Iterate up to `max_iter` examples from a batch-size-1 loader: extract the prompt up to and including `<THINK>`, sample `num_completions` tool-augmented completions with `generate_parallel_tool`, score them via `grade_responses(..., correct_weight, format_weight)`, compute `rl_loss`, backpropagate, and step the optimizer.

---

## CLI Runner (`examples/train_reasoning.rs`)

Replaces the notebook's interactive download, SFT, evaluation, and RL cells:

```bash
# 1. Sample 4 parallel completions from model_base.pth
cargo run --release -p hw7 --example train_reasoning -- --mode base --num-completions 4

# 2. Pretokenize GSM8K train & test splits
cargo run --release -p hw7 --example train_reasoning -- --mode pretokenize --limit 256

# 3. Run SFT from model_base.pth
cargo run --release -p hw7 --example train_reasoning -- --mode sft --epochs 1 --max-iter 10 --save /tmp/hw7_sft.safetensors

# 4. Evaluate model_sft.pth on GSM8K test problems
cargo run --release -p hw7 --example train_reasoning -- --mode eval --num-completions 8 --max-cases 10

# 5. Run RL from model_sft.pth
cargo run --release -p hw7 --example train_reasoning -- --mode rl --num-completions 8 --max-iter 10 --save /tmp/hw7_rl.safetensors

# 6. Solve a math question with live <TOOL> execution
cargo run --release -p hw7 --example train_reasoning -- --mode solve \
  --prompt "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
```

Set `HW7_DIR=/path/to/dir` to reuse local `model_base.pth`, `model_sft.pth`, `params.json`, `gsm8k_train.json`, and `gsm8k_test.json` files without re-downloading.
