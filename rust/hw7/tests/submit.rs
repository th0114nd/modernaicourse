//! Grader-equivalent submission tests for Homework 7.
//!
//! Ported from `homework/hw7/hw7_tests.py` (`submit_*` functions) with verified
//! ground-truth assertions plus extra edge cases:
//! - Batched KV-cache equivalence with `batch_size > 1` & cache-isolation across rows
//! - `generate_parallel` early stopping & zero-padding when all sequences hit `eot_token`
//! - GSM8K formatting edge cases (no tool calls, multiple tool calls)
//! - `eval_tool` operator precedence, negative numbers, near-integer rounding, division by zero
//! - `generate_parallel_tool` missing `<TOOL>` -> `ERROR` response & multi-tool trajectories
//! - `rl_loss` zero-variance reward handling & reference-model zero-gradient check via `LLM::deep_clone()`
//! - Checkpoint save/load round-trip & synthetic end-to-end SFT + RL training steps
//!
//! Run with: `cargo test -p hw7 --test submit`
//!
//! Full HuggingFace checkpoint evaluation test (`submit_eval_sft_pretrained`) is marked
//! `#[ignore]` so the default suite runs fast and 100% offline.

mod submit {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use candle_core::{D, DType, Device, Result, Tensor, Var, backprop::GradStore};
    use common::{assert_allclose, seeded_randn};
    use hw7::{
        ANSWER_CLOSE, ANSWER_CLOSE_ID, ANSWER_OPEN, ANSWER_OPEN_ID, Adam, CausalLM, DataLoader,
        EOT_ID, Embedding, Gsm8kExample, LLM, Linear, MultiHeadAttentionKVCache, Optimizer,
        QUESTION_CLOSE, QUESTION_CLOSE_ID, QUESTION_OPEN, QUESTION_OPEN_ID, RESPONSE_CLOSE,
        RESPONSE_CLOSE_ID, RESPONSE_OPEN, RESPONSE_OPEN_ID, ReasoningTokenizer, THINK_CLOSE,
        THINK_CLOSE_ID, THINK_OPEN, THINK_OPEN_ID, TOOL_CLOSE, TOOL_CLOSE_ID, TOOL_OPEN,
        TOOL_OPEN_ID, Tokenizer, cross_entropy_loss, download_hw7_files, eval_model, eval_tool,
        extract_answer, generate_parallel, generate_parallel_tool, get_loss_mask, grade_responses,
        gsm8k_to_text, load_pretrained_llm, log_probs, mask_indices, pretokenize_gsm8k, rl_loss,
        train_llm_rl, train_llm_sft,
    };

    static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(prefix: &str) -> Self {
            let id = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "hw7_submit_{prefix}_{}_{id}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        fn join(&self, name: &str) -> PathBuf {
            self.path.join(name)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn causal_mask(length: usize, device: &Device) -> Tensor {
        let mut data = vec![0.0_f32; length * length];
        for r in 0..length {
            for c in (r + 1)..length {
                data[r * length + c] = f32::NEG_INFINITY;
            }
        }
        Tensor::from_vec(data, (length, length), device).unwrap()
    }

    fn ref_sdpa(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let d = q.dim(D::Minus1).unwrap() as f64;
        let k_t = k
            .transpose(D::Minus2, D::Minus1)
            .unwrap()
            .contiguous()
            .unwrap();
        let mut scores = (q.contiguous().unwrap().matmul(&k_t).unwrap() / d.sqrt()).unwrap();
        if let Some(m) = mask {
            scores = scores.broadcast_add(m).unwrap();
        }
        let max = scores.max_keepdim(D::Minus1).unwrap();
        let exp = scores.broadcast_sub(&max).unwrap().exp().unwrap();
        let sum = exp.sum_keepdim(D::Minus1).unwrap();
        let probs = exp.broadcast_div(&sum).unwrap();
        probs.matmul(&v.contiguous().unwrap()).unwrap()
    }

    fn ref_mha(
        x: &Tensor,
        wq: &Tensor,
        wk: &Tensor,
        wv: &Tensor,
        wp: &Tensor,
        n_heads: usize,
        mask: Option<&Tensor>,
    ) -> Tensor {
        let (b, s, d) = x.dims3().unwrap();
        let hd = d / n_heads;
        let proj = |w: &Tensor| {
            x.broadcast_matmul(&w.t().unwrap())
                .unwrap()
                .reshape((b, s, n_heads, hd))
                .unwrap()
                .transpose(1, 2)
                .unwrap()
        };
        let q = proj(wq);
        let k = proj(wk);
        let v = proj(wv);
        let out = ref_sdpa(&q, &k, &v, mask)
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((b, s, d))
            .unwrap();
        out.broadcast_matmul(&wp.t().unwrap()).unwrap()
    }

    fn ref_log_probs(logits: &Tensor, y: &Tensor, mask: &Tensor) -> Tensor {
        let (b, s, v) = logits.dims3().unwrap();
        let z = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let targets = y.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        let masks = mask.flatten_all().unwrap().to_vec1::<u8>().unwrap();
        let mut out = Vec::with_capacity(b);
        for row_i in 0..b {
            let mut sum = 0.0_f64;
            for t in 0..s {
                let idx = row_i * s + t;
                if masks[idx] != 0 {
                    let row = &z[idx * v..(idx + 1) * v];
                    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                    let lse = max
                        + row
                            .iter()
                            .map(|&val| ((val as f64) - max).exp())
                            .sum::<f64>()
                            .ln();
                    sum += row[targets[idx] as usize] as f64 - lse;
                }
            }
            out.push(sum as f32);
        }
        Tensor::from_vec(out, (b,), logits.device()).unwrap()
    }

    fn u32_2d(rows: &[&[u32]]) -> Tensor {
        let r = rows.len();
        let c = rows[0].len();
        let flat: Vec<u32> = rows.iter().flat_map(|row| row.iter().copied()).collect();
        Tensor::from_vec(flat, (r, c), &Device::Cpu).unwrap()
    }

    fn mask_2d(rows: &[&[bool]]) -> Tensor {
        let r = rows.len();
        let c = rows[0].len();
        let flat: Vec<u8> = rows
            .iter()
            .flat_map(|row| row.iter().map(|&b| u8::from(b)))
            .collect();
        Tensor::from_vec(flat, (r, c), &Device::Cpu).unwrap()
    }

    fn pad_rows(rows: &[Vec<u32>]) -> Tensor {
        let max_len = rows.iter().map(|r| r.len()).max().unwrap_or(0);
        let mut flat = Vec::with_capacity(rows.len() * max_len);
        for r in rows {
            flat.extend_from_slice(r);
            flat.resize(flat.len() + (max_len - r.len()), 0);
        }
        Tensor::from_vec(flat, (rows.len(), max_len), &Device::Cpu).unwrap()
    }

    struct SimpleSgd {
        params: Vec<Var>,
        lr: f64,
        steps: usize,
    }

    impl SimpleSgd {
        fn new(params: Vec<Var>, lr: f64) -> Self {
            Self {
                params,
                lr,
                steps: 0,
            }
        }
    }

    impl Optimizer for SimpleSgd {
        fn step(&mut self, grads: &GradStore) -> Result<()> {
            self.steps += 1;
            for p in &self.params {
                if let Some(g) = grads.get(p.as_tensor()) {
                    p.set(&(p.as_tensor() - (g * self.lr)?)?)?;
                }
            }
            Ok(())
        }
    }

    struct FixedEncodeTokenizer {
        outputs: HashMap<String, Vec<u32>>,
        calls: RefCell<Vec<String>>,
    }

    impl FixedEncodeTokenizer {
        fn new(pairs: &[(&str, &[u32])]) -> Self {
            Self {
                outputs: pairs
                    .iter()
                    .map(|(k, v)| ((*k).to_string(), v.to_vec()))
                    .collect(),
                calls: RefCell::new(Vec::new()),
            }
        }
    }

    impl Tokenizer for FixedEncodeTokenizer {
        fn encode(&self, text: &str) -> Vec<u32> {
            self.calls.borrow_mut().push(text.to_string());
            self.outputs[text].clone()
        }

        fn decode(&self, _tokens: &[u32]) -> Result<String> {
            Ok(String::new())
        }

        fn special_token(&self, _name: &str) -> Option<u32> {
            None
        }
    }

    struct ToyGsmTokenizer;

    impl Tokenizer for ToyGsmTokenizer {
        fn encode(&self, _text: &str) -> Vec<u32> {
            Vec::new()
        }

        fn decode(&self, _tokens: &[u32]) -> Result<String> {
            Ok(String::new())
        }

        fn special_token(&self, name: &str) -> Option<u32> {
            match name {
                QUESTION_OPEN => Some(91),
                QUESTION_CLOSE => Some(92),
                THINK_OPEN => Some(93),
                THINK_CLOSE => Some(94),
                TOOL_OPEN => Some(95),
                TOOL_CLOSE => Some(96),
                RESPONSE_OPEN => Some(97),
                RESPONSE_CLOSE => Some(98),
                ANSWER_OPEN => Some(99),
                ANSWER_CLOSE => Some(100),
                _ => None,
            }
        }
    }

    struct CharSpecialTokenizer {
        special_to_id: HashMap<&'static str, u32>,
        id_to_special: HashMap<u32, &'static str>,
        char_to_id: HashMap<char, u32>,
        id_to_char: HashMap<u32, char>,
        vocab_size: usize,
    }

    impl CharSpecialTokenizer {
        fn new() -> Self {
            let specials = [
                QUESTION_OPEN,
                QUESTION_CLOSE,
                THINK_OPEN,
                THINK_CLOSE,
                TOOL_OPEN,
                TOOL_CLOSE,
                RESPONSE_OPEN,
                RESPONSE_CLOSE,
                ANSWER_OPEN,
                ANSWER_CLOSE,
            ];
            let mut special_to_id = HashMap::new();
            let mut id_to_special = HashMap::new();
            for (i, &tok) in specials.iter().enumerate() {
                let id = (i + 1) as u32;
                special_to_id.insert(tok, id);
                id_to_special.insert(id, tok);
            }
            let mut char_to_id = HashMap::new();
            let mut id_to_char = HashMap::new();
            let mut next_id = (specials.len() + 1) as u32;
            for ch in
                "0123456789+-*/(). ERRORabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ?:,\n ="
                    .chars()
            {
                if let std::collections::hash_map::Entry::Vacant(e) = char_to_id.entry(ch) {
                    e.insert(next_id);
                    id_to_char.insert(next_id, ch);
                    next_id += 1;
                }
            }
            Self {
                special_to_id,
                id_to_special,
                char_to_id,
                id_to_char,
                vocab_size: (next_id + 8) as usize,
            }
        }
    }

    impl Tokenizer for CharSpecialTokenizer {
        fn encode(&self, text: &str) -> Vec<u32> {
            let mut specials: Vec<&'static str> = self.special_to_id.keys().copied().collect();
            specials.sort_by_key(|s| std::cmp::Reverse(s.len()));

            let mut out = Vec::new();
            let mut i = 0;
            while i < text.len() {
                let rest = &text[i..];
                if let Some(&sp) = specials.iter().find(|&&s| rest.starts_with(s)) {
                    out.push(self.special_to_id[sp]);
                    i += sp.len();
                } else {
                    let ch = rest.chars().next().unwrap();
                    out.push(self.char_to_id[&ch]);
                    i += ch.len_utf8();
                }
            }
            out
        }

        fn decode(&self, tokens: &[u32]) -> Result<String> {
            let mut s = String::new();
            for &t in tokens {
                if t == 0 {
                    continue;
                }
                if let Some(&sp) = self.id_to_special.get(&t) {
                    s.push_str(sp);
                } else if let Some(&ch) = self.id_to_char.get(&t) {
                    s.push(ch);
                } else {
                    candle_core::bail!("unknown token id {t}");
                }
            }
            Ok(s)
        }

        fn special_token(&self, name: &str) -> Option<u32> {
            self.special_to_id.get(name).copied()
        }
    }

    struct BatchedToyGenerationModel {
        next_tokens: Vec<Vec<u32>>,
        vocab_size: usize,
        calls: Vec<(Vec<Vec<u32>>, usize, bool)>,
    }

    impl BatchedToyGenerationModel {
        fn new(next_tokens: Vec<Vec<u32>>, vocab_size: usize) -> Self {
            Self {
                next_tokens,
                vocab_size,
                calls: Vec::new(),
            }
        }
    }

    impl CausalLM for BatchedToyGenerationModel {
        fn forward(
            &mut self,
            tokens: &Tensor,
            seq_pos: usize,
            use_kv_cache: bool,
        ) -> Result<Tensor> {
            let step = self.calls.len();
            let row_tokens = &self.next_tokens[step];
            let (b, s) = tokens.dims2()?;
            assert_eq!(row_tokens.len(), b);
            self.calls.push((tokens.to_vec2::<u32>()?, seq_pos, use_kv_cache));
            let mut logits = vec![-1e9_f32; b * s * self.vocab_size];
            for (i, &tok) in row_tokens.iter().enumerate() {
                logits[(i * s + (s - 1)) * self.vocab_size + tok as usize] = 0.0;
            }
            Tensor::from_vec(logits, (b, s, self.vocab_size), tokens.device())
        }
    }

    struct ScriptedMultiEpisodeModel {
        episodes: Vec<Vec<Vec<u32>>>,
        placeholder: u32,
        vocab_size: usize,
        cur_episode: Option<usize>,
        cur_step: usize,
        recorded_prompts: Vec<Vec<u32>>,
        table: Var,
        train_calls: usize,
    }

    impl ScriptedMultiEpisodeModel {
        fn from_completions(
            completions_per_episode: &[Vec<Vec<u32>>],
            prompt_lens: &[usize],
            placeholder: u32,
            vocab_size: usize,
            seed: u64,
        ) -> Self {
            let mut episodes = Vec::with_capacity(completions_per_episode.len());
            for (rows, &plen) in completions_per_episode.iter().zip(prompt_lens.iter()) {
                let max_len = rows.iter().map(|r| r.len()).max().unwrap_or(plen);
                let n_steps = max_len.saturating_sub(plen);
                let mut steps = Vec::with_capacity(n_steps);
                for t in 0..n_steps {
                    let row_toks: Vec<u32> = rows
                        .iter()
                        .map(|r| r.get(plen + t).copied().unwrap_or(placeholder))
                        .collect();
                    steps.push(row_toks);
                }
                episodes.push(steps);
            }
            let init = seeded_randn(&[vocab_size, vocab_size], seed, &Device::Cpu).unwrap();
            let table = Var::from_tensor(&init).unwrap();
            Self {
                episodes,
                placeholder,
                vocab_size,
                cur_episode: None,
                cur_step: 0,
                recorded_prompts: Vec::new(),
                table,
                train_calls: 0,
            }
        }
    }

    impl CausalLM for ScriptedMultiEpisodeModel {
        fn forward(
            &mut self,
            tokens: &Tensor,
            seq_pos: usize,
            use_kv_cache: bool,
        ) -> Result<Tensor> {
            let (b, s) = tokens.dims2()?;
            if use_kv_cache {
                if seq_pos == 0 {
                    let next_ep = self.cur_episode.map_or(0, |e| e + 1);
                    self.cur_episode = Some(next_ep);
                    self.cur_step = 0;
                    let rows = tokens.to_vec2::<u32>()?;
                    self.recorded_prompts.push(rows[0].clone());
                }
                let ep = self.cur_episode.unwrap();
                let step = self.cur_step;
                self.cur_step += 1;
                let mut logits = vec![-1e9_f32; b * s * self.vocab_size];
                if let Some(row_tokens) = self.episodes[ep].get(step) {
                    assert_eq!(row_tokens.len(), b);
                    for (i, &tok) in row_tokens.iter().enumerate() {
                        logits[(i * s + (s - 1)) * self.vocab_size + tok as usize] = 0.0;
                    }
                } else {
                    for i in 0..b {
                        logits[(i * s + (s - 1)) * self.vocab_size + self.placeholder as usize] =
                            0.0;
                    }
                }
                Tensor::from_vec(logits, (b, s, self.vocab_size), tokens.device())
            } else {
                self.train_calls += 1;
                let flat = tokens.flatten_all()?;
                self.table
                    .as_tensor()
                    .index_select(&flat, 0)?
                    .reshape((b, s, self.vocab_size))
            }
        }
    }

    struct TinyTrainModel {
        embed: Embedding,
        proj: Linear,
    }

    impl TinyTrainModel {
        fn new_seeded(vocab_size: usize, dim: usize, seed: u64) -> Self {
            let device = Device::Cpu;
            let embed = Embedding::new(vocab_size, dim, &device).unwrap();
            let proj = Linear::new(dim, vocab_size, &device).unwrap();
            embed
                .weight
                .set(&seeded_randn(&[vocab_size, dim], seed, &device).unwrap())
                .unwrap();
            proj.weight
                .set(&seeded_randn(&[vocab_size, dim], seed + 1, &device).unwrap())
                .unwrap();
            Self { embed, proj }
        }

        fn clone_fresh(&self) -> Self {
            let device = self.embed.weight.device();
            let (v, d) = self.embed.weight.dims2().unwrap();
            let embed = Embedding::new(v, d, device).unwrap();
            let proj = Linear::new(d, v, device).unwrap();
            embed.weight.set(self.embed.weight.as_tensor()).unwrap();
            proj.weight.set(self.proj.weight.as_tensor()).unwrap();
            Self { embed, proj }
        }

        fn parameters(&self) -> Vec<Var> {
            vec![self.embed.weight.clone(), self.proj.weight.clone()]
        }
    }

    impl CausalLM for TinyTrainModel {
        fn forward(
            &mut self,
            tokens: &Tensor,
            _seq_pos: usize,
            _use_kv_cache: bool,
        ) -> Result<Tensor> {
            let h = self.embed.forward(tokens)?;
            self.proj.forward(&h)
        }
    }

    struct LookupLM {
        table: Var,
    }

    impl LookupLM {
        fn new(data: Vec<f32>, rows: usize, vocab: usize) -> Self {
            let table = Var::from_vec(data, (rows, vocab), &Device::Cpu).unwrap();
            Self { table }
        }
    }

    impl CausalLM for LookupLM {
        fn forward(
            &mut self,
            tokens: &Tensor,
            _seq_pos: usize,
            _use_kv_cache: bool,
        ) -> Result<Tensor> {
            let (b, s) = tokens.dims2()?;
            let v = self.table.dim(1)?;
            let flat = tokens.flatten_all()?;
            self.table
                .as_tensor()
                .index_select(&flat, 0)?
                .reshape((b, s, v))
        }
    }

    // ---------------- Q1: MultiHeadAttentionKVCache ----------------

    #[test]
    fn submit_multi_head_attention_kv_cache() {
        let device = Device::Cpu;
        let make_layer = || {
            let layer = MultiHeadAttentionKVCache::new(8, 2, 6, 3, &device).unwrap();
            let mk = |div: f32| {
                let vals: Vec<f32> = (0..64).map(|i| i as f32 / div).collect();
                Tensor::from_vec(vals, (8, 8), &device).unwrap()
            };
            layer.wq.weight.set(&mk(50.0)).unwrap();
            layer.wk.weight.set(&mk(40.0)).unwrap();
            layer.wv.weight.set(&mk(30.0)).unwrap();
            layer.wp.weight.set(&mk(20.0)).unwrap();
            layer
        };

        let x_vals: Vec<f32> = (0..(2 * 4 * 8)).map(|i| i as f32 / 10.0).collect();
        let x = Tensor::from_vec(x_vals, (2, 4, 8), &device).unwrap();
        let mask = causal_mask(4, &device);

        let mut layer = make_layer();
        let ref_out = ref_mha(
            &x,
            layer.wq.weight.as_tensor(),
            layer.wk.weight.as_tensor(),
            layer.wv.weight.as_tensor(),
            layer.wp.weight.as_tensor(),
            2,
            Some(&mask),
        );

        let full = layer.forward(&x, Some(&mask), 0, false).unwrap();
        assert_allclose(&full, &ref_out, 1e-4, 1e-4);

        let mut layer_cached = make_layer();
        let p = layer_cached
            .forward(
                &x.narrow(1, 0, 2).unwrap(),
                Some(&mask.narrow(0, 0, 2).unwrap().narrow(1, 0, 2).unwrap()),
                0,
                true,
            )
            .unwrap();
        let t = layer_cached
            .forward(
                &x.narrow(1, 2, 2).unwrap(),
                Some(&mask.narrow(0, 2, 2).unwrap()),
                2,
                true,
            )
            .unwrap();
        assert_allclose(&p, &ref_out.narrow(1, 0, 2).unwrap(), 1e-4, 1e-4);
        assert_allclose(&t, &ref_out.narrow(1, 2, 2).unwrap(), 1e-4, 1e-4);

        // Verify exact cache contents on first 2 batch rows and untouched 3rd batch row.
        let mut layer_k = make_layer();
        layer_k
            .forward(
                &x.narrow(1, 0, 3).unwrap(),
                Some(&mask.narrow(0, 0, 3).unwrap().narrow(1, 0, 3).unwrap()),
                0,
                true,
            )
            .unwrap();
        let k_expected = layer_k.wk.forward(&x.narrow(1, 0, 3).unwrap()).unwrap();
        assert_allclose(
            &layer_k.k_cache.narrow(0, 0, 2).unwrap().narrow(1, 0, 3).unwrap(),
            &k_expected,
            1e-5,
            1e-5,
        );
        let row2_sum = layer_k
            .k_cache
            .narrow(0, 2, 1)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(row2_sum, 0.0, "unused cache batch row should remain zero");

        // Edge case: batch_size > max_cache_batches must return Err.
        let x_too_big = Tensor::zeros((4, 2, 8), DType::F32, &device).unwrap();
        assert!(layer_k.forward(&x_too_big, None, 0, true).is_err());

        // Edge case: full LLM batched KV-cache equivalence with batch_size = 3.
        let mut llm = LLM::new(16, 12, 3, 10, 16, 2, &device).unwrap();
        let toks = u32_2d(&[&[1, 2, 3, 4, 5], &[5, 4, 3, 2, 1], &[2, 2, 7, 8, 9]]);
        let full_logits = llm.forward(&toks, 0, false).unwrap();
        let p_logits = llm.forward(&toks.narrow(1, 0, 3).unwrap(), 0, true).unwrap();
        let t_logits = llm.forward(&toks.narrow(1, 3, 2).unwrap(), 3, true).unwrap();
        assert_allclose(&p_logits, &full_logits.narrow(1, 0, 3).unwrap(), 1e-4, 1e-4);
        assert_allclose(&t_logits, &full_logits.narrow(1, 3, 2).unwrap(), 1e-4, 1e-4);
    }

    // ---------------- Q2: generate_parallel ----------------

    #[test]
    fn submit_generate_parallel() {
        // Exact 3 cases from hw7_tests.py::submit_generate_parallel
        let mut m1 = BatchedToyGenerationModel::new(vec![vec![3, 4], vec![5, 6]], 10);
        let t1 = generate_parallel(&mut m1, &[1], 2, None, 0.7, 3).unwrap();
        assert_eq!(t1.to_vec2::<u32>().unwrap(), vec![vec![1, 3, 5], vec![1, 4, 6]]);

        let mut m2 = BatchedToyGenerationModel::new(
            vec![vec![3, 4, 5], vec![6, 7, 6], vec![8, 6, 9]],
            12,
        );
        let t2 = generate_parallel(&mut m2, &[1, 2], 3, Some(6), 0.7, 5).unwrap();
        assert_eq!(
            t2.to_vec2::<u32>().unwrap(),
            vec![vec![1, 2, 3, 6, 8], vec![1, 2, 4, 7, 6], vec![1, 2, 5, 6, 9]]
        );

        let mut m3 = BatchedToyGenerationModel::new(vec![vec![8], vec![9], vec![7]], 12);
        let t3 = generate_parallel(&mut m3, &[2, 3], 1, Some(7), 0.7, 5).unwrap();
        assert_eq!(t3.to_vec2::<u32>().unwrap(), vec![vec![2, 3, 8, 9, 7]]);

        // Edge case: early stop well before max_tokens once ALL rows have produced eot_token.
        let mut m4 = BatchedToyGenerationModel::new(
            vec![vec![9, 4], vec![5, 9]],
            12,
        );
        let t4 = generate_parallel(&mut m4, &[1, 2], 2, Some(9), 0.7, 10).unwrap();
        assert_eq!(m4.calls.len(), 2, "must stop as soon as all completions hit eot_token");
        let rows4 = t4.to_vec2::<u32>().unwrap();
        assert_eq!(&rows4[0][..4], &[1, 2, 9, 5]);
        assert_eq!(&rows4[1][..4], &[1, 2, 4, 9]);
        assert!(rows4[0][4..].iter().all(|&v| v == 0));
        assert!(rows4[1][4..].iter().all(|&v| v == 0));
    }

    // ---------------- Q3: gsm8k_to_text ----------------

    #[test]
    fn submit_gsm8k_to_text() {
        let cases = [
            (
                Gsm8kExample::new("How much is 3 plus 4?", "Compute <<3+4=7>>.\n#### 7"),
                "<QUESTION>How much is 3 plus 4?</QUESTION><THINK>Compute <TOOL>3+4</TOOL><RESPONSE>7</RESPONSE>.</THINK><ANSWER>7</ANSWER>",
            ),
            (
                Gsm8kExample::new("No tool call here?", "Think carefully.\n#### 11"),
                "<QUESTION>No tool call here?</QUESTION><THINK>Think carefully.</THINK><ANSWER>11</ANSWER>",
            ),
            (
                Gsm8kExample::new(
                    "Two steps?",
                    "Start <<8/2=4>> and finish <<4+9=13>>.\n#### 13",
                ),
                "<QUESTION>Two steps?</QUESTION><THINK>Start <TOOL>8/2</TOOL><RESPONSE>4</RESPONSE> and finish <TOOL>4+9</TOOL><RESPONSE>13</RESPONSE>.</THINK><ANSWER>13</ANSWER>",
            ),
            // Real GSM8K Natalia example from hw7.py prose
            (
                Gsm8kExample::new(
                    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
                    "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
                ),
                "<QUESTION>Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?</QUESTION><THINK>Natalia sold 48/2 = <TOOL>48/2</TOOL><RESPONSE>24</RESPONSE>24 clips in May.\nNatalia sold 48+24 = <TOOL>48+24</TOOL><RESPONSE>72</RESPONSE>72 clips altogether in April and May.</THINK><ANSWER>72</ANSWER>",
            ),
        ];
        for (ex, expected) in cases {
            assert_eq!(gsm8k_to_text(&ex), expected);
        }
    }

    // ---------------- Q4: pretokenize_gsm8k ----------------

    #[test]
    fn submit_pretokenize_gsm8k() {
        type PretokCase<'a> = (Vec<Gsm8kExample>, Vec<(&'a str, &'a [u32])>, Vec<Vec<u32>>);
        let cases: Vec<PretokCase<'_>> = vec![
            (
                vec![
                    Gsm8kExample::new("A", "Do <<1+1=2>>.\n#### 2"),
                    Gsm8kExample::new("B", "Skip tools.\n#### 4"),
                ],
                vec![
                    ("<QUESTION>A</QUESTION><THINK>Do <TOOL>1+1</TOOL><RESPONSE>2</RESPONSE>.</THINK><ANSWER>2</ANSWER>", &[1, 2, 3]),
                    ("<QUESTION>B</QUESTION><THINK>Skip tools.</THINK><ANSWER>4</ANSWER>", &[4, 5]),
                ],
                vec![vec![1, 2, 3], vec![4, 5]],
            ),
            (
                vec![Gsm8kExample::new(
                    "C",
                    "First <<9-2=7>> then <<7*3=21>>.\n#### 21",
                )],
                vec![(
                    "<QUESTION>C</QUESTION><THINK>First <TOOL>9-2</TOOL><RESPONSE>7</RESPONSE> then <TOOL>7*3</TOOL><RESPONSE>21</RESPONSE>.</THINK><ANSWER>21</ANSWER>",
                    &[6, 7, 8, 9],
                )],
                vec![vec![6, 7, 8, 9]],
            ),
            (
                vec![
                    Gsm8kExample::new("D", "Reasoning only.\n#### 10"),
                    Gsm8kExample::new("E", "Compute <<6/3=2>>.\n#### 2"),
                ],
                vec![
                    ("<QUESTION>D</QUESTION><THINK>Reasoning only.</THINK><ANSWER>10</ANSWER>", &[10, 11]),
                    ("<QUESTION>E</QUESTION><THINK>Compute <TOOL>6/3</TOOL><RESPONSE>2</RESPONSE>.</THINK><ANSWER>2</ANSWER>", &[12, 13, 14]),
                ],
                vec![vec![10, 11], vec![12, 13, 14]],
            ),
        ];

        for (i, (examples, mapping, expected)) in cases.into_iter().enumerate() {
            let tok = FixedEncodeTokenizer::new(&mapping);
            let tmp = TempDir::new(&format!("pretok_{i}"));
            let in_p = tmp.join("gsm8k.json");
            let out_p = tmp.join("tokens.json");
            fs::write(&in_p, serde_json::to_string(&examples).unwrap()).unwrap();
            pretokenize_gsm8k(&tok, &in_p, &out_p).unwrap();
            let actual: Vec<Vec<u32>> =
                serde_json::from_str(&fs::read_to_string(&out_p).unwrap()).unwrap();
            assert_eq!(actual, expected);
        }

        // Verify ReasoningTokenizer special token IDs match hw7.py exactly (50257..=50266).
        let rtok = ReasoningTokenizer::new().unwrap();
        assert_eq!(rtok.eot_token(), EOT_ID);
        assert_eq!(rtok.special_token(QUESTION_OPEN), Some(QUESTION_OPEN_ID));
        assert_eq!(rtok.special_token(QUESTION_CLOSE), Some(QUESTION_CLOSE_ID));
        assert_eq!(rtok.special_token(THINK_OPEN), Some(THINK_OPEN_ID));
        assert_eq!(rtok.special_token(THINK_CLOSE), Some(THINK_CLOSE_ID));
        assert_eq!(rtok.special_token(TOOL_OPEN), Some(TOOL_OPEN_ID));
        assert_eq!(rtok.special_token(TOOL_CLOSE), Some(TOOL_CLOSE_ID));
        assert_eq!(rtok.special_token(RESPONSE_OPEN), Some(RESPONSE_OPEN_ID));
        assert_eq!(rtok.special_token(RESPONSE_CLOSE), Some(RESPONSE_CLOSE_ID));
        assert_eq!(rtok.special_token(ANSWER_OPEN), Some(ANSWER_OPEN_ID));
        assert_eq!(rtok.special_token(ANSWER_CLOSE), Some(ANSWER_CLOSE_ID));
        let sample = gsm8k_to_text(&Gsm8kExample::new("2+2?", "<<2+2=4>>\n#### 4"));
        let ids = rtok.encode(&sample);
        assert_eq!(ids[0], QUESTION_OPEN_ID);
        assert!(ids.contains(&TOOL_OPEN_ID));
        assert!(ids.contains(&RESPONSE_CLOSE_ID));
        assert_eq!(*ids.last().unwrap(), ANSWER_CLOSE_ID);
        assert_eq!(rtok.decode(&ids).unwrap(), sample);
    }

    // ---------------- Q5: get_loss_mask ----------------

    #[test]
    fn submit_get_loss_mask() {
        let tok = ToyGsmTokenizer;
        let sp = |name| tok.special_token(name).unwrap();

        let cases = [
            (
                vec![
                    sp(QUESTION_OPEN),
                    1,
                    sp(QUESTION_CLOSE),
                    sp(THINK_OPEN),
                    2,
                    sp(ANSWER_OPEN),
                    3,
                    sp(ANSWER_CLOSE),
                    4,
                ],
                vec![false, false, false, false, true, true, true, true, false],
            ),
            (
                vec![
                    sp(THINK_OPEN),
                    5,
                    sp(TOOL_OPEN),
                    6,
                    sp(TOOL_CLOSE),
                    sp(RESPONSE_OPEN),
                    7,
                    sp(RESPONSE_CLOSE),
                    8,
                ],
                vec![false, true, true, true, true, false, false, false, true],
            ),
            (
                vec![
                    sp(QUESTION_OPEN),
                    9,
                    sp(QUESTION_CLOSE),
                    sp(THINK_OPEN),
                    10,
                    sp(TOOL_OPEN),
                    11,
                    sp(TOOL_CLOSE),
                    sp(RESPONSE_OPEN),
                    12,
                    sp(RESPONSE_CLOSE),
                    13,
                    sp(THINK_CLOSE),
                    sp(ANSWER_OPEN),
                    14,
                    sp(ANSWER_CLOSE),
                ],
                vec![
                    false, false, false, false, true, true, true, true, false, false, false, true,
                    true, true, true, true,
                ],
            ),
        ];

        for (tokens, expected) in cases {
            assert_eq!(get_loss_mask(&tokens, &tok), expected);
        }
    }

    // ---------------- Q6: DataLoader ----------------

    #[test]
    fn submit_dataloader() {
        let tok = ToyGsmTokenizer;
        let t = tok.special_token(THINK_OPEN).unwrap();
        let a = tok.special_token(ANSWER_CLOSE).unwrap();

        let cases = [
            (
                vec![vec![1, t, 10, a], vec![2, t, 11, 12, a]],
                2_usize,
                0_usize,
                1_usize,
                vec![vec![1, t, 10, a], vec![2, t, 11, 12]],
                vec![vec![0_u8, 1, 1, 0], vec![0, 1, 1, 1]],
            ),
            (
                vec![
                    vec![3, t, 13, 14, a],
                    vec![4, t, 15, a],
                    vec![5, t, 16, 17, 18, a],
                    vec![6, t, 19, a],
                ],
                2,
                1,
                2,
                vec![vec![5, t, 16, 17, 18], vec![6, t, 19, a, 0]],
                vec![vec![0_u8, 1, 1, 1, 1], vec![0, 1, 1, 0, 0]],
            ),
            (
                vec![vec![7, t, 20, 21, 22, a]],
                1,
                0,
                1,
                vec![vec![7, t, 20, 21, 22]],
                vec![vec![0_u8, 1, 1, 1, 1]],
            ),
        ];

        for (i, (chats, bs, bidx, exp_len, exp_x, exp_m)) in cases.into_iter().enumerate() {
            let tmp = TempDir::new(&format!("dl_{i}"));
            let p = tmp.join("gsm.json");
            fs::write(&p, serde_json::to_string(&chats).unwrap()).unwrap();
            let loader = DataLoader::new(&p, bs, &tok, Device::Cpu).unwrap();
            assert_eq!(loader.len(), exp_len);
            let batches: Vec<_> = loader.iter().map(|b| b.unwrap()).collect();
            assert_eq!(batches.len(), exp_len);
            let (x, _y, m) = &batches[bidx];
            assert_eq!(x.to_vec2::<u32>().unwrap(), exp_x);
            assert_eq!(m.to_vec2::<u8>().unwrap(), exp_m);
        }

        // Edge case: batch_size = 0 should fail cleanly.
        let tmp = TempDir::new("dl_err");
        let p = tmp.join("empty.json");
        fs::write(&p, "[]").unwrap();
        assert!(DataLoader::new(&p, 0, &tok, Device::Cpu).is_err());
    }

    // ---------------- Q7: train_llm_sft ----------------

    #[test]
    fn submit_train_llm_sft() {
        let run_case = |seed: u64,
                        batches: Vec<(Tensor, Tensor, Tensor)>,
                        eval_tokens: Tensor,
                        max_iter: Option<usize>| {
            let mut model = TinyTrainModel::new_seeded(6, 4, seed);
            let mut ref_model = model.clone_fresh();

            let mut opt = SimpleSgd::new(model.parameters(), 0.1);
            train_llm_sft(&mut model, batches.iter().cloned().map(Ok), &mut opt, max_iter)
                .unwrap();

            let mut ref_opt = SimpleSgd::new(ref_model.parameters(), 0.1);
            for (step, (x, y, m)) in batches.iter().enumerate() {
                if let Some(lim) = max_iter
                    && step >= lim
                {
                    break;
                }
                let logits = ref_model.forward(x, 0, false).unwrap();
                let (b, s, v) = logits.dims3().unwrap();
                let idx = mask_indices(m).unwrap();
                let ml = logits.reshape((b * s, v)).unwrap().index_select(&idx, 0).unwrap();
                let my = y.reshape((b * s,)).unwrap().index_select(&idx, 0).unwrap();
                let grads = cross_entropy_loss(&ml, &my).unwrap().backward().unwrap();
                ref_opt.step(&grads).unwrap();
            }

            let out = model.forward(&eval_tokens, 0, false).unwrap();
            let ref_out = ref_model.forward(&eval_tokens, 0, false).unwrap();
            assert_allclose(&out, &ref_out, 1e-5, 1e-5);
        };

        run_case(
            1,
            vec![(
                u32_2d(&[&[0, 1, 2], &[1, 2, 3]]),
                u32_2d(&[&[1, 2, 3], &[2, 3, 4]]),
                mask_2d(&[&[false, true, true], &[true, false, true]]),
            )],
            u32_2d(&[&[0, 1, 2]]),
            Some(1),
        );
        run_case(
            2,
            vec![
                (
                    u32_2d(&[&[0, 1, 2], &[1, 2, 3]]),
                    u32_2d(&[&[1, 2, 3], &[2, 3, 4]]),
                    mask_2d(&[&[false, true, true], &[true, false, true]]),
                ),
                (
                    u32_2d(&[&[2, 3, 4], &[0, 2, 4]]),
                    u32_2d(&[&[3, 4, 0], &[2, 4, 1]]),
                    mask_2d(&[&[true, true, false], &[false, true, true]]),
                ),
            ],
            u32_2d(&[&[2, 3, 4]]),
            None,
        );
        run_case(
            3,
            vec![
                (
                    u32_2d(&[&[4, 3, 2], &[1, 0, 5]]),
                    u32_2d(&[&[3, 2, 1], &[0, 5, 4]]),
                    mask_2d(&[&[true, false, true], &[false, true, true]]),
                ),
                (
                    u32_2d(&[&[5, 4, 3], &[2, 1, 0]]),
                    u32_2d(&[&[4, 3, 2], &[1, 0, 5]]),
                    mask_2d(&[&[true, true, false], &[true, false, true]]),
                ),
            ],
            u32_2d(&[&[1, 0, 5]]),
            Some(2),
        );

        // Synthetic end-to-end SFT loss reduction on a tiny LLM + Adam.
        let device = Device::Cpu;
        let mut llm = LLM::new(16, 16, 2, 16, 32, 1, &device).unwrap();
        let x = u32_2d(&[&[1, 2, 3, 4, 5], &[1, 2, 6, 7, 8]]);
        let y = u32_2d(&[&[2, 3, 4, 5, 9], &[2, 6, 7, 8, 9]]);
        let m = mask_2d(&[&[false, false, true, true, true], &[false, false, true, true, true]]);
        let eval_loss = |model: &mut LLM| {
            let logits = model.forward(&x, 0, false).unwrap();
            let (b, s, v) = logits.dims3().unwrap();
            let idx = mask_indices(&m).unwrap();
            let ml = logits.reshape((b * s, v)).unwrap().index_select(&idx, 0).unwrap();
            let my = y.reshape((b * s,)).unwrap().index_select(&idx, 0).unwrap();
            cross_entropy_loss(&ml, &my).unwrap().to_scalar::<f32>().unwrap()
        };
        let loss_before = eval_loss(&mut llm);
        let mut opt = Adam::new(llm.parameters(), 2e-2, (0.9, 0.95), 1e-8).unwrap();
        let batches: Vec<_> = (0..15).map(|_| Ok((x.clone(), y.clone(), m.clone()))).collect();
        train_llm_sft(&mut llm, batches, &mut opt, None).unwrap();
        let loss_after = eval_loss(&mut llm);
        assert!(
            loss_after < loss_before * 0.5,
            "expected SFT loss to drop substantially: before={loss_before}, after={loss_after}"
        );
    }

    // ---------------- Q8: eval_tool ----------------

    #[test]
    fn submit_eval_tool() {
        assert_eq!(eval_tool("48/2"), "24");
        assert_eq!(eval_tool("7/4"), "1.75");
        assert_eq!(eval_tool("bad + 1"), "ERROR");
        // Edge cases
        assert_eq!(eval_tool("(4 + 5) * 2 - 3"), "15");
        assert_eq!(eval_tool("-10 + 3 * 2"), "-4");
        assert_eq!(eval_tool("10 / 4"), "2.5");
        assert_eq!(eval_tool("2.99995"), "3");
        assert_eq!(eval_tool("-0.00001"), "0");
        assert_eq!(eval_tool("2**3"), "8");
        assert_eq!(eval_tool(""), "ERROR");
        assert_eq!(eval_tool("5 / (2 - 2)"), "ERROR");
        assert_eq!(eval_tool("(1 + 2"), "ERROR");
    }

    // ---------------- Q9: generate_parallel_tool ----------------

    #[test]
    fn submit_generate_parallel_tool() {
        let tok = CharSpecialTokenizer::new();
        let tool_open = tok.special_token(TOOL_OPEN).unwrap();
        let tool_close = tok.special_token(TOOL_CLOSE).unwrap();
        let answer_open = tok.special_token(ANSWER_OPEN).unwrap();
        let answer_close = tok.special_token(ANSWER_CLOSE).unwrap();
        let think_close = tok.special_token(THINK_CLOSE).unwrap();
        let ch = |s: &str| tok.encode(s)[0];
        let ph = ch("x");

        // Case 1: single completion with tool call 4+5=9
        let p1 = "<QUESTION>A</QUESTION><THINK>";
        let mut m1 = BatchedToyGenerationModel::new(
            vec![
                vec![ch("4")],
                vec![ch("+")],
                vec![ch("5")],
                vec![ch("=")],
                vec![tool_open],
                vec![ch("4")],
                vec![ch("+")],
                vec![ch("5")],
                vec![tool_close],
                vec![ph],
                vec![ph],
                vec![ph],
                vec![ch(".")],
                vec![think_close],
                vec![answer_open],
                vec![ch("9")],
                vec![answer_close],
            ],
            tok.vocab_size,
        );
        let t1 =
            generate_parallel_tool(&mut m1, &tok.encode(p1), &tok, 1, Some(answer_close), 0.7, 48)
                .unwrap();
        let exp1 = tok.encode(&format!(
            "{p1}4+5=<TOOL>4+5</TOOL><RESPONSE>9</RESPONSE>.</THINK><ANSWER>9</ANSWER>"
        ));
        let r1 = t1.to_vec2::<u32>().unwrap();
        assert_eq!(&r1[0][..exp1.len()], &exp1[..]);
        assert!(r1[0][exp1.len()..].iter().all(|&v| v == 0));
        assert_eq!(m1.calls.len(), 17);

        // Case 2: division by zero -> ERROR (7 response tokens)
        let p2 = "<QUESTION>B</QUESTION><THINK>";
        let mut m2 = BatchedToyGenerationModel::new(
            vec![
                vec![tool_open],
                vec![ch("1")],
                vec![ch("/")],
                vec![ch("0")],
                vec![tool_close],
                vec![ph],
                vec![ph],
                vec![ph],
                vec![ph],
                vec![ph],
                vec![ph],
                vec![ph],
                vec![ch(".")],
                vec![think_close],
                vec![answer_open],
                vec![ch("0")],
                vec![answer_close],
            ],
            tok.vocab_size,
        );
        let t2 =
            generate_parallel_tool(&mut m2, &tok.encode(p2), &tok, 1, Some(answer_close), 0.7, 48)
                .unwrap();
        let exp2 = tok.encode(&format!(
            "{p2}<TOOL>1/0</TOOL><RESPONSE>ERROR</RESPONSE>.</THINK><ANSWER>0</ANSWER>"
        ));
        let r2 = t2.to_vec2::<u32>().unwrap();
        assert_eq!(&r2[0][..exp2.len()], &exp2[..]);
        assert!(r2[0][exp2.len()..].iter().all(|&v| v == 0));
        assert_eq!(m2.calls.len(), 17);

        // Case 3: 2 completions, one finishes early with </ANSWER> (zeroed thereafter),
        // the other invokes <TOOL>8-5</TOOL>
        let p3 = "<QUESTION>C</QUESTION><THINK>";
        let mut m3 = BatchedToyGenerationModel::new(
            vec![
                vec![ch("8"), ch("8")],
                vec![ch("-"), think_close],
                vec![ch("5"), answer_open],
                vec![ch("="), ch("8")],
                vec![tool_open, answer_close],
                vec![ch("8"), ph],
                vec![ch("-"), ph],
                vec![ch("5"), ph],
                vec![tool_close, ph],
                vec![ph, ph],
                vec![ph, ph],
                vec![ph, ph],
                vec![ch("."), ph],
                vec![think_close, ph],
                vec![answer_open, ph],
                vec![ch("3"), ph],
                vec![answer_close, ph],
            ],
            tok.vocab_size,
        );
        let t3 =
            generate_parallel_tool(&mut m3, &tok.encode(p3), &tok, 2, Some(answer_close), 0.7, 48)
                .unwrap();
        let exp3_0 = tok.encode(&format!(
            "{p3}8-5=<TOOL>8-5</TOOL><RESPONSE>3</RESPONSE>.</THINK><ANSWER>3</ANSWER>"
        ));
        let exp3_1 = tok.encode(&format!("{p3}8</THINK><ANSWER>8</ANSWER>"));
        let r3 = t3.to_vec2::<u32>().unwrap();
        assert_eq!(&r3[0][..exp3_0.len()], &exp3_0[..]);
        assert!(r3[0][exp3_0.len()..].iter().all(|&v| v == 0));
        assert_eq!(&r3[1][..exp3_1.len()], &exp3_1[..]);
        assert!(r3[1][exp3_1.len()..].iter().all(|&v| v == 0));
        assert_eq!(m3.calls.len(), 17);

        // Case 4: eot_token = </THINK> (does not zero after </THINK>, stops when both hit it)
        let p4 = "<QUESTION>D</QUESTION><THINK>";
        let mut m4 = BatchedToyGenerationModel::new(
            vec![
                vec![ch("a"), ch("b")],
                vec![think_close, ch("c")],
                vec![ph, think_close],
            ],
            tok.vocab_size,
        );
        let t4 =
            generate_parallel_tool(&mut m4, &tok.encode(p4), &tok, 2, Some(think_close), 0.7, 48)
                .unwrap();
        let exp4_0 = tok.encode(&format!("{p4}a</THINK>x"));
        let exp4_1 = tok.encode(&format!("{p4}bc</THINK>"));
        let r4 = t4.to_vec2::<u32>().unwrap();
        assert_eq!(&r4[0][..exp4_0.len()], &exp4_0[..]);
        assert!(r4[0][exp4_0.len()..].iter().all(|&v| v == 0));
        assert_eq!(&r4[1][..exp4_1.len()], &exp4_1[..]);
        assert!(r4[1][exp4_1.len()..].iter().all(|&v| v == 0));
        assert_eq!(m4.calls.len(), 3);

        // Extra edge case: stray </TOOL> without preceding <TOOL> -> <RESPONSE>ERROR</RESPONSE>
        let p5 = "<QUESTION>E</QUESTION><THINK>";
        let mut m5 = BatchedToyGenerationModel::new(
            vec![
                vec![tool_close],
                vec![ph],
                vec![ph],
                vec![ph],
                vec![ph],
                vec![ph],
                vec![ph],
                vec![ph],
                vec![answer_open],
                vec![ch("1")],
                vec![answer_close],
            ],
            tok.vocab_size,
        );
        let t5 =
            generate_parallel_tool(&mut m5, &tok.encode(p5), &tok, 1, Some(answer_close), 0.7, 32)
                .unwrap();
        let exp5 = tok.encode(&format!(
            "{p5}</TOOL><RESPONSE>ERROR</RESPONSE><ANSWER>1</ANSWER>"
        ));
        let r5 = t5.to_vec2::<u32>().unwrap();
        assert_eq!(&r5[0][..exp5.len()], &exp5[..]);
    }

    // ---------------- Q10a: extract_answer ----------------

    #[test]
    fn submit_extract_answer() {
        let tok = CharSpecialTokenizer::new();
        assert_eq!(extract_answer(&tok, &tok.encode("<ANSWER>101</ANSWER>")), Some(101));
        assert_eq!(
            extract_answer(&tok, &tok.encode("<QUESTION>Q</QUESTION><ANSWER>-4</ANSWER>")),
            Some(-4)
        );
        assert_eq!(extract_answer(&tok, &tok.encode("<ANSWER>bad</ANSWER>")), None);
        // Edge cases: whitespace inside tags, reversed tags, unclosed tag, float inside tags
        assert_eq!(extract_answer(&tok, &tok.encode("<ANSWER> 42 </ANSWER>")), Some(42));
        assert_eq!(extract_answer(&tok, &tok.encode("</ANSWER>42<ANSWER>")), None);
        assert_eq!(extract_answer(&tok, &tok.encode("<ANSWER>42")), None);
        assert_eq!(extract_answer(&tok, &tok.encode("<ANSWER>3.5</ANSWER>")), None);
    }

    // ---------------- Q10b: grade_responses ----------------

    #[test]
    fn submit_grade_responses() {
        let tok = CharSpecialTokenizer::new();
        let check = |rows: &[&str], gt: &str, cw: f64, fw: f64, expected: &[f64]| {
            let encoded: Vec<Vec<u32>> = rows.iter().map(|s| tok.encode(s)).collect();
            let t = pad_rows(&encoded);
            let gt_v = tok.encode(gt);
            let gt_t = Tensor::from_vec(gt_v.clone(), (gt_v.len(),), &Device::Cpu).unwrap();
            let scores = grade_responses(&tok, &t, &gt_t, cw, fw).unwrap();
            assert_eq!(scores.len(), expected.len());
            for (a, b) in scores.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-9, "got {a}, expected {b}");
            }
        };

        check(
            &["<ANSWER>5</ANSWER>", "<ANSWER>3</ANSWER>"],
            "<ANSWER>5</ANSWER>",
            1.0,
            0.1,
            &[1.1, 0.1],
        );
        check(
            &[
                "<THINK>x</THINK>",
                "<ANSWER>-2</ANSWER>",
                "<ANSWER>-2</ANSWER>",
            ],
            "<ANSWER>-2</ANSWER>",
            2.0,
            0.5,
            &[0.0, 2.5, 2.5],
        );
        check(
            &["<ANSWER>8</ANSWER>", "<ANSWER>oops</ANSWER>"],
            "<ANSWER>7</ANSWER>",
            1.5,
            0.25,
            &[0.25, 0.0],
        );
    }

    // ---------------- Q11: eval_model ----------------

    #[test]
    fn submit_eval_model() {
        let tok = CharSpecialTokenizer::new();
        let p1 = "<QUESTION>A</QUESTION><THINK>";
        let p2 = "<QUESTION>B</QUESTION><THINK>";
        let p3 = "<QUESTION>C</QUESTION><THINK>";
        let p1_ids = tok.encode(p1);
        let p2_ids = tok.encode(p2);
        let p3_ids = tok.encode(p3);

        let mk_batch = |prompt: &[u32], suffix: &str, gt: &str| {
            let mut x = prompt.to_vec();
            x.extend(tok.encode(suffix));
            Ok((
                u32_2d(&[&x]),
                u32_2d(&[&tok.encode(gt)]),
                Tensor::zeros((1, x.len()), DType::U8, &Device::Cpu).unwrap(),
            ))
        };

        // Case 1: 1 example, 2 completions (1 correct, 2 formatted) -> (0.5, 1.0, 1.0)
        let loader1 = vec![mk_batch(&p1_ids, "x", "<ANSWER>2</ANSWER>")];
        let mut m1 = ScriptedMultiEpisodeModel::from_completions(
            &[vec![
                tok.encode(&format!(
                    "{p1}1+1=<TOOL>1+1</TOOL><RESPONSE>2</RESPONSE>.</THINK><ANSWER>2</ANSWER>"
                )),
                tok.encode(&format!("{p1}9.</THINK><ANSWER>9</ANSWER>")),
            ]],
            &[p1_ids.len()],
            tok.encode("x")[0],
            tok.vocab_size,
            10,
        );
        let (a1, f1, pk1) = eval_model(loader1, &mut m1, &tok, 2, 30, 0.6, 1).unwrap();
        assert!((a1 - 0.5).abs() < 1e-6);
        assert!((f1 - 1.0).abs() < 1e-6);
        assert!((pk1 - 1.0).abs() < 1e-6);

        // Case 2: 3 examples in loader, max_cases = 2 -> (0.5, 0.75, 1.0)
        let loader2 = vec![
            mk_batch(&p1_ids, "x", "<ANSWER>2</ANSWER>"),
            mk_batch(&p2_ids, "y", "<ANSWER>3</ANSWER>"),
            mk_batch(&p3_ids, "z", "<ANSWER>4</ANSWER>"),
        ];
        let mut m2 = ScriptedMultiEpisodeModel::from_completions(
            &[
                vec![
                    tok.encode(&format!("{p1}2.</THINK><ANSWER>2</ANSWER>")),
                    tok.encode(&format!("{p1}bad")),
                ],
                vec![
                    tok.encode(&format!("{p2}1.</THINK><ANSWER>1</ANSWER>")),
                    tok.encode(&format!("{p2}3.</THINK><ANSWER>3</ANSWER>")),
                ],
                vec![
                    tok.encode(&format!("{p3}bad")),
                    tok.encode(&format!("{p3}bad")),
                ],
            ],
            &[p1_ids.len(), p2_ids.len(), p3_ids.len()],
            tok.encode("x")[0],
            tok.vocab_size,
            11,
        );
        let (a2, f2, pk2) = eval_model(loader2, &mut m2, &tok, 2, 20, 0.6, 2).unwrap();
        assert!((a2 - 0.5).abs() < 1e-6);
        assert!((f2 - 0.75).abs() < 1e-6);
        assert!((pk2 - 1.0).abs() < 1e-6);
        assert_eq!(m2.recorded_prompts.len(), 2);

        // Case 3: max_cases = 1, num_completions = 1 -> (1.0, 1.0, 1.0)
        let loader3 = vec![
            mk_batch(&p1_ids, "x", "<ANSWER>2</ANSWER>"),
            mk_batch(&p2_ids, "y", "<ANSWER>3</ANSWER>"),
        ];
        let mut m3 = ScriptedMultiEpisodeModel::from_completions(
            &[
                vec![tok.encode(&format!("{p1}2.</THINK><ANSWER>2</ANSWER>"))],
                vec![tok.encode(&format!("{p2}3.</THINK><ANSWER>3</ANSWER>"))],
            ],
            &[p1_ids.len(), p2_ids.len()],
            tok.encode("x")[0],
            tok.vocab_size,
            12,
        );
        let (a3, f3, pk3) = eval_model(loader3, &mut m3, &tok, 1, 20, 0.6, 1).unwrap();
        assert!((a3 - 1.0).abs() < 1e-6);
        assert!((f3 - 1.0).abs() < 1e-6);
        assert!((pk3 - 1.0).abs() < 1e-6);
    }

    // ---------------- Q12: rl_loss ----------------

    #[test]
    fn submit_rl_loss() {
        let tok = CharSpecialTokenizer::new();
        let cases = [
            (
                [
                    "<QUESTION>Q</QUESTION><THINK>a</THINK><ANSWER>5</ANSWER>",
                    "<QUESTION>R</QUESTION><THINK>b</THINK><ANSWER>2</ANSWER>",
                ],
                vec![1.0_f32, 0.0],
            ),
            (
                [
                    "<QUESTION>X</QUESTION><THINK>c</THINK><ANSWER>9</ANSWER>",
                    "<QUESTION>Y</QUESTION><THINK>d</THINK><ANSWER>4</ANSWER>",
                ],
                vec![0.25_f32, 1.75],
            ),
            (
                [
                    "<QUESTION>M</QUESTION><THINK>e</THINK><ANSWER>1</ANSWER>",
                    "<QUESTION>N</QUESTION><THINK>f</THINK><ANSWER>3</ANSWER>",
                ],
                vec![-1.0_f32, 2.0],
            ),
        ];

        for (text_rows, rewards_vec) in cases {
            let r0 = tok.encode(text_rows[0]);
            let r1 = tok.encode(text_rows[1]);
            let tokens = u32_2d(&[&r0, &r1]);
            let vocab_size = (*r0.iter().chain(r1.iter()).max().unwrap() as usize + 5).max(40);
            let table_data: Vec<f32> = (0..(vocab_size * vocab_size))
                .map(|i| i as f32 / 50.0)
                .collect();
            let mut model = LookupLM::new(table_data, vocab_size, vocab_size);
            let rewards = Tensor::from_vec(rewards_vec.clone(), (2,), &Device::Cpu).unwrap();

            let loss = rl_loss(&mut model, &tok, &tokens, &rewards).unwrap();

            let m0 = get_loss_mask(&r0, &tok);
            let m1 = get_loss_mask(&r1, &tok);
            let mask = mask_2d(&[&m0, &m1]);
            let s = tokens.dim(1).unwrap();
            let logits = model
                .forward(&tokens.narrow(1, 0, s - 1).unwrap(), 0, false)
                .unwrap();
            let logp = ref_log_probs(
                &logits,
                &tokens.narrow(1, 1, s - 1).unwrap(),
                &mask.narrow(1, 1, s - 1).unwrap(),
            );
            let mean_r = rewards.mean_all().unwrap();
            let adv = rewards.broadcast_sub(&mean_r).unwrap();
            let n_tok: usize = m0.iter().chain(m1.iter()).filter(|&&b| b).count();
            let expected =
                ((logp * adv).unwrap().sum_all().unwrap().neg().unwrap() / n_tok as f64).unwrap();
            assert_allclose(&loss, &expected, 1e-5, 1e-5);
        }

        // Edge case: zero-variance rewards (all equal) must yield exactly 0.0 (no NaN!).
        let r0 = tok.encode("<QUESTION>Q</QUESTION><THINK>a</THINK><ANSWER>5</ANSWER>");
        let r1 = tok.encode("<QUESTION>R</QUESTION><THINK>b</THINK><ANSWER>2</ANSWER>");
        let tokens = u32_2d(&[&r0, &r1]);
        let mut model = LookupLM::new(vec![0.1; 100 * 100], 100, 100);
        let eq_rewards = Tensor::from_vec(vec![0.5_f32, 0.5], (2,), &Device::Cpu).unwrap();
        let zero_loss = rl_loss(&mut model, &tok, &tokens, &eq_rewards).unwrap();
        let val = zero_loss.to_scalar::<f32>().unwrap();
        assert!(val.is_finite() && val.abs() < 1e-6);

        // Edge case: deep_clone reference model gets ZERO gradients & remains untouched.
        let device = Device::Cpu;
        let mut policy = LLM::new(tok.vocab_size, 16, 2, 32, 32, 1, &device).unwrap();
        let policy_ref = policy.deep_clone().unwrap();
        let ref_before = policy_ref.output.weight.as_tensor().copy().unwrap();
        let rewards = Tensor::from_vec(vec![1.0_f32, 0.0], (2,), &device).unwrap();
        let loss = rl_loss(&mut policy, &tok, &tokens, &rewards).unwrap();
        let grads = loss.backward().unwrap();
        assert!(grads.get(policy.output.weight.as_tensor()).is_some());
        assert!(
            grads.get(policy_ref.output.weight.as_tensor()).is_none(),
            "frozen deep_clone reference model must receive no gradients"
        );
        let mut opt = Adam::new(policy.parameters(), 1e-2, (0.9, 0.95), 1e-8).unwrap();
        opt.step(&grads).unwrap();
        assert_allclose(policy_ref.output.weight.as_tensor(), &ref_before, 0.0, 0.0);

        // Bonus check: save_weights -> load_weights round-trip on LLM.
        let tmp = TempDir::new("ckpt");
        let ckpt_path = tmp.join("policy.safetensors");
        policy.save_weights(&ckpt_path).unwrap();
        let mut reloaded = LLM::new(tok.vocab_size, 16, 2, 32, 32, 1, &device).unwrap();
        reloaded.load_weights(&ckpt_path).unwrap();
        let out_a = policy.forward(&tokens, 0, false).unwrap();
        let out_b = reloaded.forward(&tokens, 0, false).unwrap();
        assert_allclose(&out_a, &out_b, 1e-6, 1e-6);
    }

    // ---------------- Q13: train_llm_rl ----------------

    #[test]
    fn submit_train_llm_rl() {
        let tok = CharSpecialTokenizer::new();
        let p1 = "<QUESTION>A</QUESTION><THINK>";
        let p2 = "<QUESTION>B</QUESTION><THINK>";
        let p1_ids = tok.encode(p1);
        let p2_ids = tok.encode(p2);

        let verify_rl_case = |episodes: Vec<Vec<Vec<u32>>>,
                              prompt_lens: Vec<usize>,
                              ground_truths: Vec<&str>,
                              max_iter: Option<usize>,
                              seed: u64| {
            let max_tokens = episodes
                .iter()
                .flatten()
                .map(|r| r.len())
                .max()
                .unwrap();
            let num_completions = episodes[0].len();

            let make_loader = || {
                episodes
                    .iter()
                    .zip(prompt_lens.iter())
                    .zip(ground_truths.iter())
                    .enumerate()
                    .map(|(i, ((ep, &plen), &gt))| {
                        let mut x = ep[0][..plen].to_vec();
                        x.push(tok.encode(&((b'a' + i as u8) as char).to_string())[0]);
                        Ok((
                            u32_2d(&[&x]),
                            u32_2d(&[&tok.encode(gt)]),
                            Tensor::zeros((1, x.len()), DType::U8, &Device::Cpu).unwrap(),
                        ))
                    })
                    .collect::<Vec<_>>()
            };

            let mut model = ScriptedMultiEpisodeModel::from_completions(
                &episodes,
                &prompt_lens,
                tok.encode("x")[0],
                tok.vocab_size,
                seed,
            );
            let mut ref_model = ScriptedMultiEpisodeModel::from_completions(
                &episodes,
                &prompt_lens,
                tok.encode("x")[0],
                tok.vocab_size,
                seed,
            );

            let mut opt = SimpleSgd::new(vec![model.table.clone()], 0.1);
            train_llm_rl(
                &mut model,
                make_loader(),
                &mut opt,
                &tok,
                num_completions,
                0.8,
                max_tokens,
                max_iter,
                1.0,
                0.25,
            )
            .unwrap();

            // Reference step-by-step execution
            let mut ref_opt = SimpleSgd::new(vec![ref_model.table.clone()], 0.1);
            let answer_close = tok.special_token(ANSWER_CLOSE);
            let expected_steps = max_iter.map_or(episodes.len(), |lim| lim.min(episodes.len()));
            for step in 0..expected_steps {
                let prompt = &episodes[step][0][..prompt_lens[step]];
                let full_toks = generate_parallel_tool(
                    &mut ref_model,
                    prompt,
                    &tok,
                    num_completions,
                    answer_close,
                    0.8,
                    max_tokens,
                )
                .unwrap();
                let gt_v = tok.encode(ground_truths[step]);
                let gt_t = Tensor::from_vec(gt_v.clone(), (gt_v.len(),), &Device::Cpu).unwrap();
                let scores = grade_responses(&tok, &full_toks, &gt_t, 1.0, 0.25).unwrap();
                let r_f32: Vec<f32> = scores.iter().map(|&v| v as f32).collect();
                let rewards = Tensor::from_vec(r_f32, (num_completions,), &Device::Cpu).unwrap();
                let loss = rl_loss(&mut ref_model, &tok, &full_toks, &rewards).unwrap();
                let grads = loss.backward().unwrap();
                ref_opt.step(&grads).unwrap();
            }

            assert_eq!(opt.steps, expected_steps);
            assert_allclose(
                model.table.as_tensor(),
                ref_model.table.as_tensor(),
                1e-5,
                1e-5,
            );
        };

        // Case 1 (exact match to hw7_tests.py::submit_train_llm_rl case 1)
        verify_rl_case(
            vec![vec![
                tok.encode(&format!(
                    "{p1}2+3=<TOOL>2+3</TOOL><RESPONSE>5</RESPONSE>.</THINK><ANSWER>5</ANSWER>"
                )),
                tok.encode(&format!("{p1}4.</THINK><ANSWER>4</ANSWER>")),
            ]],
            vec![p1_ids.len()],
            vec!["<ANSWER>5</ANSWER>"],
            Some(1),
            20,
        );

        // Case 2 (exact match to hw7_tests.py::submit_train_llm_rl case 2)
        verify_rl_case(
            vec![
                vec![
                    tok.encode(&format!(
                        "{p1}2+3=<TOOL>2+3</TOOL><RESPONSE>5</RESPONSE>.</THINK><ANSWER>5</ANSWER>"
                    )),
                    tok.encode(&format!("{p1}bad")),
                ],
                vec![
                    tok.encode(&format!(
                        "{p2}1+2=<TOOL>1+2</TOOL><RESPONSE>3</RESPONSE>.</THINK><ANSWER>3</ANSWER>"
                    )),
                    tok.encode(&format!("{p2}1.</THINK><ANSWER>1</ANSWER>")),
                ],
            ],
            vec![p1_ids.len(), p2_ids.len()],
            vec!["<ANSWER>5</ANSWER>", "<ANSWER>3</ANSWER>"],
            None,
            21,
        );

        // Case 3 (exact match to hw7_tests.py::submit_train_llm_rl case 3)
        verify_rl_case(
            vec![
                vec![
                    tok.encode(&format!(
                        "{p1}5+3=<TOOL>5+3</TOOL><RESPONSE>8</RESPONSE>.</THINK><ANSWER>8</ANSWER>"
                    )),
                    tok.encode(&format!("{p1}2.</THINK><ANSWER>2</ANSWER>")),
                ],
                vec![
                    tok.encode(&format!("{p2}oops")),
                    tok.encode(&format!(
                        "{p2}3+3=<TOOL>3+3</TOOL><RESPONSE>6</RESPONSE>.</THINK><ANSWER>6</ANSWER>"
                    )),
                ],
            ],
            vec![p1_ids.len(), p2_ids.len()],
            vec!["<ANSWER>8</ANSWER>", "<ANSWER>6</ANSWER>"],
            Some(2),
            22,
        );

        // Verify policy-gradient direction: repeated rl_loss updates on a real LLM increase
        // log-probability of the high-reward completion relative to the low-reward completion.
        let device = Device::Cpu;
        let mut llm = LLM::new(tok.vocab_size, 16, 2, 32, 32, 1, &device).unwrap();
        let good = tok.encode("<QUESTION>Q</QUESTION><THINK>ok</THINK><ANSWER>5</ANSWER>");
        let bad = tok.encode("<QUESTION>Q</QUESTION><THINK>no</THINK><ANSWER>1</ANSWER>");
        let toks = u32_2d(&[&good, &bad]);
        let rewards = Tensor::from_vec(vec![1.0_f32, 0.0], (2,), &device).unwrap();
        let m_good = get_loss_mask(&good, &tok);
        let m_bad = get_loss_mask(&bad, &tok);
        let mask_shift = mask_2d(&[&m_good[1..], &m_bad[1..]]);
        let x_in = toks.narrow(1, 0, good.len() - 1).unwrap();
        let y_in = toks.narrow(1, 1, good.len() - 1).unwrap();

        let margin = |m: &mut LLM| {
            let l = m.forward(&x_in, 0, false).unwrap();
            let lp = log_probs(&l, &y_in, &mask_shift).unwrap().to_vec1::<f32>().unwrap();
            lp[0] - lp[1]
        };
        let margin_before = margin(&mut llm);
        let mut opt = Adam::new(llm.parameters(), 1e-2, (0.9, 0.95), 1e-8).unwrap();
        for _ in 0..8 {
            let loss = rl_loss(&mut llm, &tok, &toks, &rewards).unwrap();
            let grads = loss.backward().unwrap();
            opt.step(&grads).unwrap();
        }
        let margin_after = margin(&mut llm);
        assert!(
            margin_after > margin_before + 0.5,
            "RL updates should increase logp(good) - logp(bad): before={margin_before}, after={margin_after}"
        );
    }

    #[test]
    #[ignore = "Downloads model_sft.pth and GSM8K from HuggingFace/GitHub; run with --ignored"]
    fn submit_eval_sft_pretrained() {
        let device = Device::Cpu;
        let files = download_hw7_files().unwrap();
        let tok = ReasoningTokenizer::new().unwrap();
        let mut model = load_pretrained_llm(&files.params, &files.model_sft, &device).unwrap();

        let tmp = TempDir::new("sft_eval_submit");
        let test_tok_path = tmp.join("gsm8k_test_tokenized.json");
        let raw: Vec<Gsm8kExample> =
            serde_json::from_str(&fs::read_to_string(&files.gsm8k_test).unwrap()).unwrap();
        let subset_in = tmp.join("gsm8k_test_subset.json");
        fs::write(&subset_in, serde_json::to_string(&raw[..8]).unwrap()).unwrap();
        pretokenize_gsm8k(&tok, &subset_in, &test_tok_path).unwrap();

        let loader = DataLoader::new(&test_tok_path, 1, &tok, device).unwrap();
        let (acc, fmt, passk) = eval_model(&loader, &mut model, &tok, 8, 180, 0.7, 4).unwrap();
        assert!((0.0..=1.0).contains(&acc));
        assert!(fmt > 0.25, "pretrained SFT model should format >25% of answers, got {fmt}");
        assert!((0.0..=1.0).contains(&passk));
    }
}
