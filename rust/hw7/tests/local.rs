//! Local sanity-check tests for Homework 7.
//!
//! Ported directly from `homework/hw7/hw7_tests.py` (`test_*` functions).
//! Run with: `cargo test -p hw7 --test local`
//!
//! Note: `test_eval_sft_pretrained` downloads `model_sft.pth`, `params.json`, and GSM8K
//! from HuggingFace Hub (`zkolter/RL-Homework`) / `openai/grade-school-math`, so it is
//! marked `#[ignore]` to keep the default test suite fast and 100% offline.
//! Run it explicitly with:
//! `cargo test --release -p hw7 --test local test_eval_sft_pretrained -- --ignored`

mod local {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use candle_core::{D, DType, Device, Result, Tensor, TensorId, Var, backprop::GradStore};
    use common::{assert_allclose, seeded_randn};
    use hw7::{
        ANSWER_CLOSE, ANSWER_OPEN, Adam, CausalLM, DataLoader, Embedding, Gsm8kExample, LLM,
        Linear, MLP, MultiHeadAttentionKVCache, Optimizer, QUESTION_CLOSE, QUESTION_OPEN,
        RESPONSE_CLOSE, RESPONSE_OPEN, ReasoningTokenizer, THINK_CLOSE, THINK_OPEN, TOOL_CLOSE,
        TOOL_OPEN, Tokenizer, TransformerBlock, cross_entropy_loss, download_hw7_files, eval_model,
        eval_tool, extract_answer, generate_parallel, generate_parallel_tool, get_loss_mask,
        grade_responses, gsm8k_to_text, load_pretrained_llm, log_probs, mask_indices,
        pretokenize_gsm8k, rl_loss, rms_norm, self_attention, silu, train_llm_rl, train_llm_sft,
    };

    static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(prefix: &str) -> Self {
            let id = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "hw7_local_{prefix}_{}_{id}",
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

    fn ref_rms_norm(x: &Tensor, eps: f64) -> Tensor {
        let dims = x.dims();
        let d = *dims.last().unwrap();
        let flat = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut out = Vec::with_capacity(flat.len());
        for row in flat.chunks_exact(d) {
            let mean_sq = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / d as f64;
            let denom = (mean_sq + eps).sqrt();
            for &v in row {
                out.push((v as f64 / denom) as f32);
            }
        }
        Tensor::from_vec(out, dims, x.device()).unwrap()
    }

    fn ref_silu(x: &Tensor) -> Tensor {
        let vals = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let out: Vec<f32> = vals
            .into_iter()
            .map(|v| {
                let vf = v as f64;
                (vf / (1.0 + (-vf).exp())) as f32
            })
            .collect();
        Tensor::from_vec(out, x.shape(), x.device()).unwrap()
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

    fn ref_mlp(mlp: &MLP, x: &Tensor) -> Tensor {
        let h = x
            .broadcast_matmul(&mlp.w1.weight.as_tensor().t().unwrap())
            .unwrap();
        ref_silu(&h)
            .broadcast_matmul(&mlp.w2.weight.as_tensor().t().unwrap())
            .unwrap()
    }

    fn ref_block(block: &TransformerBlock, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let x_norm = ref_rms_norm(x, 1e-5);
        let attn = ref_mha(
            &x_norm,
            block.attn.wq.weight.as_tensor(),
            block.attn.wk.weight.as_tensor(),
            block.attn.wv.weight.as_tensor(),
            block.attn.wp.weight.as_tensor(),
            block.attn.n_heads,
            mask,
        );
        let z = (x + attn).unwrap();
        let z_norm = ref_rms_norm(&z, 1e-5);
        (&z + ref_mlp(&block.mlp, &z_norm)).unwrap()
    }

    fn ref_llm(model: &LLM, tokens: &Tensor) -> Tensor {
        let seq_len = tokens.dim(1).unwrap();
        let x_emb = model.embedding.forward(tokens).unwrap();
        let pos = model
            .pos_embeddings
            .as_tensor()
            .narrow(0, 0, seq_len)
            .unwrap();
        let mut x = x_emb.broadcast_add(&pos).unwrap();
        let mask = model
            .mask
            .narrow(0, 0, seq_len)
            .unwrap()
            .narrow(1, 0, seq_len)
            .unwrap();
        for layer in &model.layers {
            x = ref_block(layer, &x, Some(&mask));
        }
        let x = ref_rms_norm(&x, 1e-5);
        x.broadcast_matmul(&model.output.weight.as_tensor().t().unwrap())
            .unwrap()
    }

    fn ref_cross_entropy(logits: &Tensor, y: &Tensor) -> f32 {
        let k = logits.dim(D::Minus1).unwrap();
        let m = y.elem_count();
        let z = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let targets = y.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        let mut total = 0.0_f64;
        for (i, &t) in targets.iter().enumerate() {
            let row = &z[i * k..(i + 1) * k];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum_exp: f64 = row.iter().map(|&v| ((v as f64) - max).exp()).sum();
            let lse = max + sum_exp.ln();
            total += lse - row[t as usize] as f64;
        }
        (total / m as f64) as f32
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

    struct RefAdam {
        params: Vec<Var>,
        lr: f64,
        betas: (f64, f64),
        eps: f64,
        m: HashMap<TensorId, Tensor>,
        v: HashMap<TensorId, Tensor>,
        t: usize,
    }

    impl RefAdam {
        fn new(params: Vec<Var>, lr: f64, betas: (f64, f64), eps: f64) -> Self {
            let m = params
                .iter()
                .map(|p| (p.id(), Tensor::zeros_like(p.as_tensor()).unwrap()))
                .collect();
            let v = params
                .iter()
                .map(|p| (p.id(), Tensor::zeros_like(p.as_tensor()).unwrap()))
                .collect();
            Self {
                params,
                lr,
                betas,
                eps,
                m,
                v,
                t: 0,
            }
        }

        fn step(&mut self, grads: &GradStore) {
            self.t += 1;
            let (b1, b2) = self.betas;
            let bc1 = 1.0 - b1.powi(self.t as i32);
            let bc2 = 1.0 - b2.powi(self.t as i32);
            for p in &self.params {
                if let Some(g) = grads.get(p.as_tensor()) {
                    let id = p.id();
                    let m =
                        ((self.m[&id].clone() * b1).unwrap() + (g * (1.0 - b1)).unwrap()).unwrap();
                    let v = ((self.v[&id].clone() * b2).unwrap()
                        + (g.sqr().unwrap() * (1.0 - b2)).unwrap())
                    .unwrap();
                    let mh = (&m / bc1).unwrap();
                    let vh = (&v / bc2).unwrap();
                    let upd = ((mh / (vh.sqrt().unwrap() + self.eps).unwrap()).unwrap() * self.lr)
                        .unwrap();
                    p.set(&(p.as_tensor() - upd).unwrap()).unwrap();
                    self.m.insert(id, m);
                    self.v.insert(id, v);
                }
            }
        }
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
            let outputs = pairs
                .iter()
                .map(|(k, v)| ((*k).to_string(), v.to_vec()))
                .collect();
            Self {
                outputs,
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

    /// Character-level + special-token tokenizer matching `_CharSpecialTokenizer` in `hw7_tests.py`.
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
            assert!(
                step < self.next_tokens.len(),
                "unexpected extra forward call at step {step}"
            );
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

    /// Multi-episode model: deterministic scripted generation when `use_kv_cache == true`,
    /// and differentiable lookup-table logits when `use_kv_cache == false`.
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

    // ---------------- Part 0: Carried-Over Model & Training Components ----------------

    #[test]
    fn test_linear() {
        let device = Device::Cpu;
        let layer = Linear::new(10, 20, &device).unwrap();
        assert_eq!(layer.weight.dims(), &[20, 10]);

        let w = seeded_randn(&[20, 10], 0, &device).unwrap();
        layer.weight.set(&w).unwrap();
        let w_t = w.t().unwrap();

        let x2d = seeded_randn(&[50, 10], 1, &device).unwrap();
        let out2d = layer.forward(&x2d).unwrap();
        assert_eq!(out2d.dims(), &[50, 20]);
        assert_allclose(&out2d, &x2d.matmul(&w_t).unwrap(), 1e-5, 1e-5);

        let x3d = seeded_randn(&[7, 9, 10], 2, &device).unwrap();
        let out3d = layer.forward(&x3d).unwrap();
        assert_eq!(out3d.dims(), &[7, 9, 20]);
        let ref3d = x3d
            .reshape((63, 10))
            .unwrap()
            .matmul(&w_t)
            .unwrap()
            .reshape((7, 9, 20))
            .unwrap();
        assert_allclose(&out3d, &ref3d, 1e-5, 1e-5);

        let init_layer = Linear::new(100, 1000, &device).unwrap();
        let vals = init_layer
            .weight
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mean = vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64;
        let var = vals
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / vals.len() as f64;
        let expected_std = (2.0_f64 / 100.0).sqrt();
        assert!((var.sqrt() - expected_std).abs() < 5e-3);
    }

    #[test]
    fn test_embedding() {
        let device = Device::Cpu;
        let layer = Embedding::new(200, 20, &device).unwrap();
        assert_eq!(layer.weight.dims(), &[200, 20]);

        let w = seeded_randn(&[200, 20], 10, &device).unwrap();
        layer.weight.set(&w).unwrap();
        let w_rows = w.to_vec2::<f32>().unwrap();

        let idx1d: Vec<u32> = (0..13).map(|i| (i * 17 + 3) % 200).collect();
        let y1d = Tensor::from_slice(&idx1d, (13,), &device).unwrap();
        let out1d = layer.forward(&y1d).unwrap();
        assert_eq!(out1d.dims(), &[13, 20]);
        let ref1d_data: Vec<f32> = idx1d
            .iter()
            .flat_map(|&i| w_rows[i as usize].iter().copied())
            .collect();
        assert_allclose(
            &out1d,
            &Tensor::from_vec(ref1d_data, (13, 20), &device).unwrap(),
            1e-6,
            1e-6,
        );

        let idx2d: Vec<u32> = (0..77).map(|i| (i * 29 + 5) % 200).collect();
        let y2d = Tensor::from_slice(&idx2d, (7, 11), &device).unwrap();
        let out2d = layer.forward(&y2d).unwrap();
        assert_eq!(out2d.dims(), &[7, 11, 20]);
        let ref2d_data: Vec<f32> = idx2d
            .iter()
            .flat_map(|&i| w_rows[i as usize].iter().copied())
            .collect();
        assert_allclose(
            &out2d,
            &Tensor::from_vec(ref2d_data, (7, 11, 20), &device).unwrap(),
            1e-6,
            1e-6,
        );

        let init_layer = Embedding::new(1000, 100, &device).unwrap();
        let vals = init_layer
            .weight
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mean = vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64;
        let var = vals
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / vals.len() as f64;
        assert!((var.sqrt() - 1.0).abs() < 3e-2);
    }

    #[test]
    fn test_silu() {
        let device = Device::Cpu;
        let x2d = seeded_randn(&[10, 20], 20, &device).unwrap();
        assert_allclose(&silu(&x2d).unwrap(), &ref_silu(&x2d), 1e-5, 1e-6);

        let x4d = seeded_randn(&[3, 4, 5, 6], 21, &device).unwrap();
        assert_allclose(&silu(&x4d).unwrap(), &ref_silu(&x4d), 1e-5, 1e-6);
    }

    #[test]
    fn test_rms_norm() {
        let device = Device::Cpu;
        let x2d = seeded_randn(&[100, 20], 30, &device).unwrap();
        assert_allclose(
            &rms_norm(&x2d, 1e-5).unwrap(),
            &ref_rms_norm(&x2d, 1e-5),
            1e-5,
            1e-5,
        );

        let x3d = seeded_randn(&[10, 7, 20], 31, &device).unwrap();
        assert_allclose(
            &rms_norm(&x3d, 1e-3).unwrap(),
            &ref_rms_norm(&x3d, 1e-3),
            1e-5,
            1e-5,
        );
    }

    #[test]
    fn test_self_attention() {
        let device = Device::Cpu;
        let mask = causal_mask(5, &device);

        let q = seeded_randn(&[5, 8], 40, &device).unwrap();
        let k = seeded_randn(&[5, 8], 41, &device).unwrap();
        let v = seeded_randn(&[5, 6], 42, &device).unwrap();
        let out = self_attention(&q, &k, &v, Some(&mask)).unwrap();
        assert_eq!(out.dims(), &[5, 6]);
        assert_allclose(&out, &ref_sdpa(&q, &k, &v, Some(&mask)), 1e-5, 1e-5);

        let q4 = seeded_randn(&[2, 3, 5, 8], 43, &device).unwrap();
        let k4 = seeded_randn(&[2, 3, 5, 8], 44, &device).unwrap();
        let v4 = seeded_randn(&[2, 3, 5, 4], 45, &device).unwrap();
        let out4 = self_attention(&q4, &k4, &v4, Some(&mask)).unwrap();
        assert_eq!(out4.dims(), &[2, 3, 5, 4]);
        assert_allclose(&out4, &ref_sdpa(&q4, &k4, &v4, Some(&mask)), 1e-5, 1e-5);
    }

    #[test]
    fn test_mlp() {
        let device = Device::Cpu;
        let mlp = MLP::new(5, 7, &device).unwrap();
        assert_eq!(mlp.w1.weight.dims(), &[7, 5]);
        assert_eq!(mlp.w2.weight.dims(), &[5, 7]);

        mlp.w1
            .weight
            .set(&seeded_randn(&[7, 5], 60, &device).unwrap())
            .unwrap();
        mlp.w2
            .weight
            .set(&seeded_randn(&[5, 7], 61, &device).unwrap())
            .unwrap();

        let x = seeded_randn(&[4, 3, 5], 62, &device).unwrap();
        let ref_out = ref_mlp(&mlp, &x);
        let out = mlp.forward(&x).unwrap();
        assert_eq!(out.dims(), &[4, 3, 5]);
        assert_allclose(&out, &ref_out, 1e-5, 1e-5);
    }

    #[test]
    fn test_transformer_block() {
        let device = Device::Cpu;
        let mut block = TransformerBlock::new(12, 3, 16, 8, &device).unwrap();
        let x = seeded_randn(&[1, 5, 12], 70, &device).unwrap();
        let mask = causal_mask(5, &device);

        let ref_out = ref_block(&block, &x, Some(&mask));
        let out = block.forward(&x, Some(&mask), 0, false).unwrap();
        assert_eq!(out.dims(), &[1, 5, 12]);
        assert_allclose(&out, &ref_out, 3e-5, 3e-5);

        block
            .forward(
                &x.narrow(1, 0, 3).unwrap(),
                Some(&mask.narrow(0, 0, 3).unwrap().narrow(1, 0, 3).unwrap()),
                0,
                true,
            )
            .unwrap();
        let tail = block
            .forward(
                &x.narrow(1, 3, 2).unwrap(),
                Some(&mask.narrow(0, 3, 2).unwrap()),
                3,
                true,
            )
            .unwrap();
        assert_allclose(&tail, &out.narrow(1, 3, 2).unwrap(), 3e-5, 3e-5);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let device = Device::Cpu;
        let logits = seeded_randn(&[100, 50], 80, &device).unwrap();
        let y_data: Vec<u32> = (0..100).map(|i| (i * 13 + 7) % 50).collect();
        let y = Tensor::from_vec(y_data, (100,), &device).unwrap();
        let mine = cross_entropy_loss(&logits, &y).unwrap();
        assert_eq!(mine.rank(), 0);
        let ref_val = ref_cross_entropy(&logits, &y);
        assert!((mine.to_scalar::<f32>().unwrap() - ref_val).abs() < 1e-5);

        let logits3 = seeded_randn(&[4, 5, 7], 81, &device).unwrap();
        let y3_data: Vec<u32> = (0..20).map(|i| (i * 5 + 1) % 7).collect();
        let y3 = Tensor::from_vec(y3_data, (4, 5), &device).unwrap();
        let mine3 = cross_entropy_loss(&logits3, &y3).unwrap();
        let ref_val3 = ref_cross_entropy(&logits3, &y3);
        assert!((mine3.to_scalar::<f32>().unwrap() - ref_val3).abs() < 1e-5);
    }

    #[test]
    fn test_adam() {
        let device = Device::Cpu;
        let l1 = Linear::new(6, 4, &device).unwrap();
        let l2 = Linear::new(4, 3, &device).unwrap();
        let l1_ref = Linear::new(6, 4, &device).unwrap();
        let l2_ref = Linear::new(4, 3, &device).unwrap();
        l1_ref.weight.set(l1.weight.as_tensor()).unwrap();
        l2_ref.weight.set(l2.weight.as_tensor()).unwrap();

        let mut opt = Adam::new(
            vec![l1.weight.clone(), l2.weight.clone()],
            1e-3,
            (0.9, 0.95),
            1e-8,
        )
        .unwrap();
        let mut ref_opt = RefAdam::new(
            vec![l1_ref.weight.clone(), l2_ref.weight.clone()],
            1e-3,
            (0.9, 0.95),
            1e-8,
        );

        for step in 0..5_u64 {
            let x = seeded_randn(&[16, 6], 100 + step, &device).unwrap();
            let y_data: Vec<u32> = (0..16).map(|i| ((i as u64 + step) % 3) as u32).collect();
            let y = Tensor::from_vec(y_data, (16,), &device).unwrap();

            let logits = l2.forward(&l1.forward(&x).unwrap().tanh().unwrap()).unwrap();
            let loss = cross_entropy_loss(&logits, &y).unwrap();
            let grads = loss.backward().unwrap();
            opt.step(&grads).unwrap();

            let r_logits = l2_ref
                .forward(&l1_ref.forward(&x).unwrap().tanh().unwrap())
                .unwrap();
            let r_loss = cross_entropy_loss(&r_logits, &y).unwrap();
            let r_grads = r_loss.backward().unwrap();
            ref_opt.step(&r_grads);
        }

        assert_allclose(l1.weight.as_tensor(), l1_ref.weight.as_tensor(), 1e-5, 1e-5);
        assert_allclose(l2.weight.as_tensor(), l2_ref.weight.as_tensor(), 1e-5, 1e-5);
    }

    #[test]
    fn test_llm() {
        let device = Device::Cpu;
        let mut model = LLM::new(11, 12, 3, 8, 16, 2, &device).unwrap();
        assert_eq!(model.embedding.weight.dims(), &[11, 12]);
        assert_eq!(model.pos_embeddings.dims(), &[8, 12]);
        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.output.weight.dims(), &[11, 12]);
        assert_eq!(model.mask.dims(), &[8, 8]);
        let mask_rows = model.mask.to_vec2::<f32>().unwrap();
        assert_eq!(mask_rows[0][0], 0.0);
        assert!(mask_rows[0][1].is_infinite() && mask_rows[0][1].is_sign_negative());
        assert_eq!(mask_rows[1][0], 0.0);

        let tokens = u32_2d(&[&[0, 1, 2, 3, 4]]);
        let ref_out = ref_llm(&model, &tokens);
        let out = model.forward(&tokens, 0, false).unwrap();
        assert_eq!(out.dims(), &[1, 5, 11]);
        assert_allclose(&out, &ref_out, 6e-5, 6e-5);

        model
            .forward(&tokens.narrow(1, 0, 3).unwrap(), 0, true)
            .unwrap();
        let tail = model
            .forward(&tokens.narrow(1, 3, 2).unwrap(), 3, true)
            .unwrap();
        assert_allclose(&tail, &out.narrow(1, 3, 2).unwrap(), 6e-5, 6e-5);
    }

    #[test]
    fn test_log_probs() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(
            vec![
                2.0_f32, 0.0, -1.0, 0.5, 1.5, -0.5, 1.0, -1.0, 0.0, // row 0
                -0.5, 1.0, 0.0, 2.0, 0.0, -2.0, 0.25, 0.25, 0.25, // row 1
            ],
            (2, 3, 3),
            &device,
        )
        .unwrap();
        let y = u32_2d(&[&[0, 1, 2], &[1, 0, 2]]);
        let mask = mask_2d(&[&[true, false, true], &[true, true, false]]);
        let out = log_probs(&logits, &y, &mask).unwrap();
        assert_eq!(out.dims(), &[2]);
        let expected = Tensor::from_vec(vec![-1.5775_f32, -0.6073], (2,), &device).unwrap();
        assert_allclose(&out, &expected, 1e-4, 1e-4);
    }

    // ---------------- Part I: Parallel Sampling ----------------

    #[test]
    fn test_multi_head_attention_kv_cache() {
        let device = Device::Cpu;
        let mut attn = MultiHeadAttentionKVCache::new(12, 3, 8, 4, &device).unwrap();
        assert_eq!(attn.k_cache.dims(), &[4, 8, 12]);
        assert_eq!(attn.v_cache.dims(), &[4, 8, 12]);

        let wq = seeded_randn(&[12, 12], 130, &device).unwrap();
        let wk = seeded_randn(&[12, 12], 131, &device).unwrap();
        let wv = seeded_randn(&[12, 12], 132, &device).unwrap();
        let wp = seeded_randn(&[12, 12], 133, &device).unwrap();
        attn.wq.weight.set(&wq).unwrap();
        attn.wk.weight.set(&wk).unwrap();
        attn.wv.weight.set(&wv).unwrap();
        attn.wp.weight.set(&wp).unwrap();

        let x = seeded_randn(&[2, 5, 12], 134, &device).unwrap();
        let mask = causal_mask(5, &device);
        let ref_out = ref_mha(&x, &wq, &wk, &wv, &wp, 3, Some(&mask));

        let out = attn.forward(&x, Some(&mask), 0, false).unwrap();
        assert_eq!(out.dims(), &[2, 5, 12]);
        assert_allclose(&out, &ref_out, 1e-5, 1e-5);

        let prefix = attn
            .forward(
                &x.narrow(1, 0, 3).unwrap(),
                Some(&mask.narrow(0, 0, 3).unwrap().narrow(1, 0, 3).unwrap()),
                0,
                true,
            )
            .unwrap();
        let tail = attn
            .forward(
                &x.narrow(1, 3, 2).unwrap(),
                Some(&mask.narrow(0, 3, 2).unwrap()),
                3,
                true,
            )
            .unwrap();
        assert_allclose(&prefix, &ref_out.narrow(1, 0, 3).unwrap(), 1e-5, 1e-5);
        assert_allclose(&tail, &ref_out.narrow(1, 3, 2).unwrap(), 1e-5, 1e-5);

        let k_expected = attn.wk.forward(&x).unwrap();
        let v_expected = attn.wv.forward(&x).unwrap();
        assert_allclose(
            &attn.k_cache.narrow(0, 0, 2).unwrap().narrow(1, 0, 5).unwrap(),
            &k_expected,
            1e-5,
            1e-5,
        );
        assert_allclose(
            &attn.v_cache.narrow(0, 0, 2).unwrap().narrow(1, 0, 5).unwrap(),
            &v_expected,
            1e-5,
            1e-5,
        );
    }

    #[test]
    fn test_generate_parallel() {
        let mut model = BatchedToyGenerationModel::new(
            vec![vec![3, 4, 5], vec![6, 8, 6], vec![9, 6, 7]],
            12,
        );

        let tokens = generate_parallel(&mut model, &[1, 2], 3, Some(6), 0.7, 5).unwrap();
        assert_eq!(
            tokens.to_vec2::<u32>().unwrap(),
            vec![vec![1, 2, 3, 6, 9], vec![1, 2, 4, 8, 6], vec![1, 2, 5, 6, 7]]
        );

        assert_eq!(model.calls.len(), 3);
        assert_eq!(model.calls[0].0, vec![vec![1, 2], vec![1, 2], vec![1, 2]]);
        assert_eq!(model.calls[0].1, 0);
        assert!(model.calls[0].2);
        assert_eq!(model.calls[1].0, vec![vec![3], vec![4], vec![5]]);
        assert_eq!(model.calls[1].1, 2);
        assert!(model.calls[1].2);
        assert_eq!(model.calls[2].0, vec![vec![6], vec![8], vec![6]]);
        assert_eq!(model.calls[2].1, 3);
        assert!(model.calls[2].2);

        let mut model2 = BatchedToyGenerationModel::new(vec![vec![4, 5], vec![6, 7]], 10);
        let tokens2 = generate_parallel(&mut model2, &[1], 2, None, 0.4, 3).unwrap();
        assert_eq!(
            tokens2.to_vec2::<u32>().unwrap(),
            vec![vec![1, 4, 6], vec![1, 5, 7]]
        );
    }

    // ---------------- Part II: Supervised Reasoning Finetuning ----------------

    #[test]
    fn test_gsm8k_to_text() {
        let example = Gsm8kExample::new(
            "What is 6 plus 7?",
            "First compute <<6+7=13>>.\nThen compute <<13*2=26>>.\n#### 26",
        );
        let expected = "<QUESTION>What is 6 plus 7?</QUESTION>\
                        <THINK>First compute <TOOL>6+7</TOOL><RESPONSE>13</RESPONSE>.\n\
                        Then compute <TOOL>13*2</TOOL><RESPONSE>26</RESPONSE>.</THINK>\
                        <ANSWER>26</ANSWER>";
        assert_eq!(gsm8k_to_text(&example), expected);
    }

    #[test]
    fn test_pretokenize_gsm8k() {
        let examples = vec![
            Gsm8kExample::new("Q1", "Use <<2+3=5>>.\n#### 5"),
            Gsm8kExample::new("Q2", "No tool.\n#### 9"),
        ];
        let text0 =
            "<QUESTION>Q1</QUESTION><THINK>Use <TOOL>2+3</TOOL><RESPONSE>5</RESPONSE>.</THINK><ANSWER>5</ANSWER>";
        let text1 = "<QUESTION>Q2</QUESTION><THINK>No tool.</THINK><ANSWER>9</ANSWER>";
        let tokenizer = FixedEncodeTokenizer::new(&[(text0, &[11, 12, 13]), (text1, &[21, 22])]);

        let tmp = TempDir::new("pretok");
        let in_path = tmp.join("gsm8k.json");
        let out_path = tmp.join("tokens.json");
        fs::write(&in_path, serde_json::to_string(&examples).unwrap()).unwrap();

        pretokenize_gsm8k(&tokenizer, &in_path, &out_path).unwrap();

        let tokens: Vec<Vec<u32>> =
            serde_json::from_str(&fs::read_to_string(&out_path).unwrap()).unwrap();
        assert_eq!(
            tokenizer.calls.borrow().as_slice(),
            &[text0.to_string(), text1.to_string()]
        );
        assert_eq!(tokens, vec![vec![11, 12, 13], vec![21, 22]]);
    }

    #[test]
    fn test_get_loss_mask() {
        let tok = ToyGsmTokenizer;
        let sp = |name| tok.special_token(name).unwrap();
        let tokens = vec![
            sp(QUESTION_OPEN),
            1,
            sp(QUESTION_CLOSE),
            sp(THINK_OPEN),
            2,
            sp(TOOL_OPEN),
            3,
            sp(TOOL_CLOSE),
            sp(RESPONSE_OPEN),
            4,
            sp(RESPONSE_CLOSE),
            5,
            sp(THINK_CLOSE),
            sp(ANSWER_OPEN),
            6,
            sp(ANSWER_CLOSE),
            7,
        ];
        let expected = vec![
            false, false, false, false, true, true, true, true, false, false, false, true, true,
            true, true, true, false,
        ];
        assert_eq!(get_loss_mask(&tokens, &tok), expected);
    }

    #[test]
    fn test_dataloader() {
        let tok = ToyGsmTokenizer;
        let think = tok.special_token(THINK_OPEN).unwrap();
        let ans_close = tok.special_token(ANSWER_CLOSE).unwrap();

        let chats = vec![
            vec![1, think, 10, 11, ans_close],
            vec![2, think, 12, ans_close],
            vec![3, think, 13, 14, 15, ans_close],
            vec![4, think, 16, ans_close],
            vec![5, think, 17, 18, ans_close],
        ];

        let tmp = TempDir::new("dataloader");
        let filename = tmp.join("gsm_tokens.json");
        fs::write(&filename, serde_json::to_string(&chats).unwrap()).unwrap();

        let loader = DataLoader::new(&filename, 2, &tok, Device::Cpu).unwrap();
        assert_eq!(loader.len(), 2);
        let batches: Vec<_> = loader.iter().map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 2);

        let (x0, y0, m0) = &batches[0];
        assert_eq!(x0.to_vec2::<u32>().unwrap(), vec![vec![1, 93, 10, 11], vec![2, 93, 12, 100]]);
        assert_eq!(y0.to_vec2::<u32>().unwrap(), vec![vec![93, 10, 11, 100], vec![93, 12, 100, 0]]);
        assert_eq!(m0.to_vec2::<u8>().unwrap(), vec![vec![0, 1, 1, 1], vec![0, 1, 1, 0]]);

        let (x1, y1, m1) = &batches[1];
        assert_eq!(
            x1.to_vec2::<u32>().unwrap(),
            vec![vec![3, 93, 13, 14, 15], vec![4, 93, 16, 100, 0]]
        );
        assert_eq!(
            y1.to_vec2::<u32>().unwrap(),
            vec![vec![93, 13, 14, 15, 100], vec![93, 16, 100, 0, 0]]
        );
        assert_eq!(
            m1.to_vec2::<u8>().unwrap(),
            vec![vec![0, 1, 1, 1, 1], vec![0, 1, 1, 0, 0]]
        );

        let batches2: Vec<_> = (&loader).into_iter().map(|b| b.unwrap()).collect();
        assert_eq!(batches2.len(), batches.len());
        for ((xa, ya, ma), (xb, yb, mb)) in batches.iter().zip(batches2.iter()) {
            assert_eq!(xa.to_vec2::<u32>().unwrap(), xb.to_vec2::<u32>().unwrap());
            assert_eq!(ya.to_vec2::<u32>().unwrap(), yb.to_vec2::<u32>().unwrap());
            assert_eq!(ma.to_vec2::<u8>().unwrap(), mb.to_vec2::<u8>().unwrap());
        }
    }

    #[test]
    fn test_train_llm_sft() {
        let batches = vec![
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
        ];

        let mut model = TinyTrainModel::new_seeded(6, 4, 0);
        let init_embed = model.embed.weight.as_tensor().copy().unwrap();
        let init_proj = model.proj.weight.as_tensor().copy().unwrap();
        let mut ref_model = model.clone_fresh();

        let mut opt = SimpleSgd::new(model.parameters(), 0.1);
        train_llm_sft(
            &mut model,
            batches.iter().cloned().map(Ok),
            &mut opt,
            Some(1),
        )
        .unwrap();
        assert_eq!(opt.steps, 1);
        let changed = (model.embed.weight.as_tensor() - &init_embed)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            > 1e-6
            || (model.proj.weight.as_tensor() - &init_proj)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                > 1e-6;
        assert!(changed);

        // Verify exact match against manual 1-step masked CE SGD.
        let mut ref_opt = SimpleSgd::new(ref_model.parameters(), 0.1);
        let (x0, y0, m0) = &batches[0];
        let logits0 = ref_model.forward(x0, 0, false).unwrap();
        let (b, s, v) = logits0.dims3().unwrap();
        let idx = mask_indices(m0).unwrap();
        let ml = logits0.reshape((b * s, v)).unwrap().index_select(&idx, 0).unwrap();
        let my = y0.reshape((b * s,)).unwrap().index_select(&idx, 0).unwrap();
        let grads = cross_entropy_loss(&ml, &my).unwrap().backward().unwrap();
        ref_opt.step(&grads).unwrap();
        assert_allclose(
            model.embed.weight.as_tensor(),
            ref_model.embed.weight.as_tensor(),
            1e-5,
            1e-5,
        );
        assert_allclose(
            model.proj.weight.as_tensor(),
            ref_model.proj.weight.as_tensor(),
            1e-5,
            1e-5,
        );

        // Full pass (max_iter = None) runs both batches.
        let mut model2 = TinyTrainModel::new_seeded(6, 4, 0);
        let mut opt2 = SimpleSgd::new(model2.parameters(), 0.1);
        train_llm_sft(&mut model2, batches.into_iter().map(Ok), &mut opt2, None).unwrap();
        assert_eq!(opt2.steps, 2);
    }

    // ---------------- Part III: Evaluating Tools & Reasoning Models ----------------

    #[test]
    fn test_eval_tool() {
        assert_eq!(eval_tool("6/3"), "2");
        assert_eq!(eval_tool("7/2"), "3.5");
        assert_eq!(eval_tool("1/0"), "ERROR");
        assert_eq!(eval_tool("3.00001"), "3");
    }

    #[test]
    fn test_generate_parallel_tool() {
        let tok = CharSpecialTokenizer::new();
        let prompt_text = "<QUESTION>Q</QUESTION><THINK>";
        let prompt = tok.encode(prompt_text);
        let tool_open = tok.special_token(TOOL_OPEN).unwrap();
        let tool_close = tok.special_token(TOOL_CLOSE).unwrap();
        let answer_open = tok.special_token(ANSWER_OPEN).unwrap();
        let answer_close = tok.special_token(ANSWER_CLOSE).unwrap();
        let think_close = tok.special_token(THINK_CLOSE).unwrap();
        let ch = |s: &str| tok.encode(s)[0];
        let placeholder = ch("x");

        let mut model = BatchedToyGenerationModel::new(
            vec![
                vec![ch("2"), ch("7")],
                vec![ch("+"), think_close],
                vec![ch("3"), answer_open],
                vec![ch("="), ch("7")],
                vec![tool_open, answer_close],
                vec![ch("2"), placeholder],
                vec![ch("+"), placeholder],
                vec![ch("3"), placeholder],
                vec![tool_close, placeholder],
                vec![placeholder, placeholder],
                vec![placeholder, placeholder],
                vec![placeholder, placeholder],
                vec![ch("."), placeholder],
                vec![think_close, placeholder],
                vec![answer_open, placeholder],
                vec![ch("5"), placeholder],
                vec![answer_close, placeholder],
            ],
            tok.vocab_size,
        );

        let tokens = generate_parallel_tool(
            &mut model,
            &prompt,
            &tok,
            2,
            Some(answer_close),
            0.7,
            40,
        )
        .unwrap();

        let row1 = tok.encode(&format!(
            "{prompt_text}2+3=<TOOL>2+3</TOOL><RESPONSE>5</RESPONSE>.</THINK><ANSWER>5</ANSWER>"
        ));
        let row2 = tok.encode(&format!("{prompt_text}7</THINK><ANSWER>7</ANSWER>"));
        let expected = pad_rows(&[row1.clone(), row2.clone()]);
        let exp_len = expected.dim(1).unwrap();
        let tok_rows = tokens.to_vec2::<u32>().unwrap();
        let exp_rows = expected.to_vec2::<u32>().unwrap();

        assert_eq!(&tok_rows[0][..exp_len], &exp_rows[0][..]);
        assert_eq!(&tok_rows[1][..exp_len], &exp_rows[1][..]);
        assert!(tok_rows[1][row2.len()..].iter().all(|&t| t == 0));
        assert!(tok_rows[0][exp_len..].iter().all(|&t| t == 0));
        assert_eq!(model.calls.len(), 17);
        assert_eq!(model.calls[0].0, vec![prompt.clone(), prompt.clone()]);
        assert_eq!(model.calls[0].1, 0);
        assert!(model.calls[0].2);

        // Verify eot_token is respected as passed (e.g., </THINK>, not hardcoded </ANSWER>).
        let mut model2 = BatchedToyGenerationModel::new(
            vec![
                vec![ch("a"), ch("b")],
                vec![think_close, ch("c")],
                vec![placeholder, think_close],
            ],
            tok.vocab_size,
        );
        let tokens2 = generate_parallel_tool(
            &mut model2,
            &prompt,
            &tok,
            2,
            Some(think_close),
            0.7,
            20,
        )
        .unwrap();
        let expected2 = pad_rows(&[
            tok.encode(&format!("{prompt_text}a</THINK>x")),
            tok.encode(&format!("{prompt_text}bc</THINK>")),
        ]);
        let exp_len2 = expected2.dim(1).unwrap();
        let tok_rows2 = tokens2.to_vec2::<u32>().unwrap();
        let exp_rows2 = expected2.to_vec2::<u32>().unwrap();
        assert_eq!(&tok_rows2[0][..exp_len2], &exp_rows2[0][..]);
        assert_eq!(&tok_rows2[1][..exp_len2], &exp_rows2[1][..]);
        assert!(tok_rows2[0][exp_len2..].iter().all(|&t| t == 0));
        assert!(tok_rows2[1][exp_len2..].iter().all(|&t| t == 0));
        assert_eq!(model2.calls.len(), 3);
    }

    #[test]
    fn test_extract_answer() {
        let tok = CharSpecialTokenizer::new();
        assert_eq!(
            extract_answer(&tok, &tok.encode("<THINK>x</THINK><ANSWER>12</ANSWER>")),
            Some(12)
        );
        assert_eq!(extract_answer(&tok, &tok.encode("<ANSWER>-7</ANSWER>")), Some(-7));
        assert_eq!(extract_answer(&tok, &tok.encode("<ANSWER>oops</ANSWER>")), None);
        assert_eq!(
            extract_answer(&tok, &tok.encode("<THINK>no answer</THINK>")),
            None
        );
    }

    #[test]
    fn test_grade_responses() {
        let tok = CharSpecialTokenizer::new();
        let rows = pad_rows(&[
            tok.encode("<THINK>a</THINK><ANSWER>9</ANSWER>"),
            tok.encode("<THINK>b</THINK><ANSWER>4</ANSWER>"),
            tok.encode("<THINK>c</THINK>"),
        ]);
        let gt_vec = tok.encode("<ANSWER>9</ANSWER>");
        let gt = Tensor::from_vec(gt_vec.clone(), (gt_vec.len(),), &Device::Cpu).unwrap();
        let scores = grade_responses(&tok, &rows, &gt, 1.0, 0.2).unwrap();
        assert_eq!(scores.len(), 3);
        assert!((scores[0] - 1.2).abs() < 1e-9);
        assert!((scores[1] - 0.2).abs() < 1e-9);
        assert!((scores[2] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_eval_model() {
        let tok = CharSpecialTokenizer::new();
        let prompt1 = tok.encode("<QUESTION>Q1</QUESTION><THINK>");
        let prompt2 = tok.encode("<QUESTION>Q2</QUESTION><THINK>");

        let mut x1 = prompt1.clone();
        x1.extend(tok.encode("abc"));
        let mut x2 = prompt2.clone();
        x2.extend(tok.encode("xyz"));

        let loader = vec![
            Ok((
                u32_2d(&[&x1]),
                u32_2d(&[&tok.encode("<ANSWER>5</ANSWER>")]),
                Tensor::zeros((1, x1.len()), DType::U8, &Device::Cpu).unwrap(),
            )),
            Ok((
                u32_2d(&[&x2]),
                u32_2d(&[&tok.encode("<ANSWER>3</ANSWER>")]),
                Tensor::zeros((1, x2.len()), DType::U8, &Device::Cpu).unwrap(),
            )),
        ];

        let ep1 = vec![
            tok.encode(
                "<QUESTION>Q1</QUESTION><THINK>2+3=<TOOL>2+3</TOOL><RESPONSE>5</RESPONSE>.</THINK><ANSWER>5</ANSWER>",
            ),
            tok.encode("<QUESTION>Q1</QUESTION><THINK>4.</THINK><ANSWER>4</ANSWER>"),
            tok.encode("<QUESTION>Q1</QUESTION><THINK>oops"),
        ];
        let ep2 = vec![
            tok.encode("<QUESTION>Q2</QUESTION><THINK>1.</THINK><ANSWER>1</ANSWER>"),
            tok.encode("<QUESTION>Q2</QUESTION><THINK>bad"),
            tok.encode("<QUESTION>Q2</QUESTION><THINK>still bad"),
        ];

        let mut model = ScriptedMultiEpisodeModel::from_completions(
            &[ep1, ep2],
            &[prompt1.len(), prompt2.len()],
            tok.encode("x")[0],
            tok.vocab_size,
            1,
        );

        let (correct, formatted, passk) =
            eval_model(loader, &mut model, &tok, 3, 25, 0.6, 2).unwrap();
        assert!((correct - 1.0 / 6.0).abs() < 1e-6);
        assert!((formatted - 0.5).abs() < 1e-6);
        assert!((passk - 0.5).abs() < 1e-6);
        assert_eq!(model.recorded_prompts, vec![prompt1, prompt2]);
    }

    // ---------------- Part IV: Reinforcement Learning ----------------

    #[test]
    fn test_rl_loss() {
        let tok = CharSpecialTokenizer::new();
        let r0 = tok.encode("<QUESTION>Q</QUESTION><THINK>a</THINK><ANSWER>5</ANSWER>");
        let r1 = tok.encode("<QUESTION>R</QUESTION><THINK>b</THINK><ANSWER>2</ANSWER>");
        assert_eq!(r0.len(), r1.len());
        let tokens = u32_2d(&[&r0, &r1]);
        let rewards = Tensor::from_vec(vec![1.5_f32, -0.5], (2,), &Device::Cpu).unwrap();

        let max_tok = r0.iter().chain(r1.iter()).copied().max().unwrap() as usize;
        let vocab_size = (max_tok + 3).max(32);
        let n_elem = vocab_size * vocab_size;
        let table_data: Vec<f32> = (0..n_elem)
            .map(|i| -1.0 + 2.0 * (i as f32) / ((n_elem - 1) as f32))
            .collect();
        let mut model = LookupLM::new(table_data, vocab_size, vocab_size);

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
        let ref_loss = ((logp * adv).unwrap().sum_all().unwrap().neg().unwrap() / n_tok as f64)
            .unwrap();
        assert_allclose(&loss, &ref_loss, 1e-5, 1e-5);

        let grads = loss.backward().unwrap();
        assert!(grads.get(model.table.as_tensor()).is_some());
    }

    #[test]
    fn test_train_llm_rl() {
        let tok = CharSpecialTokenizer::new();
        let prompt1 = tok.encode("<QUESTION>Q1</QUESTION><THINK>");
        let prompt2 = tok.encode("<QUESTION>Q2</QUESTION><THINK>");
        let gt1 = tok.encode("<ANSWER>5</ANSWER>");
        let gt2 = tok.encode("<ANSWER>3</ANSWER>");

        let ep1 = vec![
            tok.encode(
                "<QUESTION>Q1</QUESTION><THINK>2+3=<TOOL>2+3</TOOL><RESPONSE>5</RESPONSE>.</THINK><ANSWER>5</ANSWER>",
            ),
            tok.encode("<QUESTION>Q1</QUESTION><THINK>4.</THINK><ANSWER>4</ANSWER>"),
        ];
        let ep2 = vec![
            tok.encode("<QUESTION>Q2</QUESTION><THINK>bad"),
            tok.encode(
                "<QUESTION>Q2</QUESTION><THINK>1+2=<TOOL>1+2</TOOL><RESPONSE>3</RESPONSE>.</THINK><ANSWER>3</ANSWER>",
            ),
        ];

        let mut x1 = prompt1.clone();
        x1.extend(tok.encode("x"));
        let mut x2 = prompt2.clone();
        x2.extend(tok.encode("y"));

        let make_loader = || {
            vec![
                Ok((
                    u32_2d(&[&x1]),
                    u32_2d(&[&gt1]),
                    Tensor::zeros((1, x1.len()), DType::U8, &Device::Cpu).unwrap(),
                )),
                Ok((
                    u32_2d(&[&x2]),
                    u32_2d(&[&gt2]),
                    Tensor::zeros((1, x2.len()), DType::U8, &Device::Cpu).unwrap(),
                )),
            ]
        };

        let max_tokens = ep1
            .iter()
            .chain(ep2.iter())
            .map(|r| r.len())
            .max()
            .unwrap();

        // max_iter = Some(1)
        let mut model1 = ScriptedMultiEpisodeModel::from_completions(
            &[ep1.clone(), ep2.clone()],
            &[prompt1.len(), prompt2.len()],
            tok.encode("x")[0],
            tok.vocab_size,
            7,
        );
        let init_table = model1.table.as_tensor().copy().unwrap();
        let mut opt1 = SimpleSgd::new(vec![model1.table.clone()], 0.1);
        train_llm_rl(
            &mut model1,
            make_loader(),
            &mut opt1,
            &tok,
            2,
            0.8,
            max_tokens,
            Some(1),
            1.0,
            0.25,
        )
        .unwrap();
        assert_eq!(opt1.steps, 1);
        assert_eq!(model1.train_calls, 1);
        assert_eq!(model1.recorded_prompts, vec![prompt1.clone()]);
        let diff = (model1.table.as_tensor() - &init_table)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff > 1e-6, "RL step should update model parameters");

        // max_iter = None (processes both examples)
        let mut model2 = ScriptedMultiEpisodeModel::from_completions(
            &[ep1, ep2],
            &[prompt1.len(), prompt2.len()],
            tok.encode("x")[0],
            tok.vocab_size,
            7,
        );
        let mut opt2 = SimpleSgd::new(vec![model2.table.clone()], 0.1);
        train_llm_rl(
            &mut model2,
            make_loader(),
            &mut opt2,
            &tok,
            2,
            0.8,
            max_tokens,
            None,
            1.0,
            0.25,
        )
        .unwrap();
        assert_eq!(opt2.steps, 2);
        assert_eq!(model2.train_calls, 2);
        assert_eq!(model2.recorded_prompts, vec![prompt1, prompt2]);
    }

    #[test]
    #[ignore = "Downloads model_sft.pth and GSM8K from HuggingFace/GitHub; run with --ignored"]
    fn test_eval_sft_pretrained() {
        let device = Device::Cpu;
        let files = download_hw7_files().unwrap();
        let tok = ReasoningTokenizer::new().unwrap();
        let mut model = load_pretrained_llm(&files.params, &files.model_sft, &device).unwrap();

        let tmp = TempDir::new("sft_eval");
        let test_tok_path = tmp.join("gsm8k_test_tokenized.json");
        let raw: Vec<Gsm8kExample> =
            serde_json::from_str(&fs::read_to_string(&files.gsm8k_test).unwrap()).unwrap();
        let subset_in = tmp.join("gsm8k_test_subset.json");
        fs::write(&subset_in, serde_json::to_string(&raw[..4]).unwrap()).unwrap();
        pretokenize_gsm8k(&tok, &subset_in, &test_tok_path).unwrap();

        let loader = DataLoader::new(&test_tok_path, 1, &tok, device).unwrap();
        let (acc, fmt, passk) = eval_model(&loader, &mut model, &tok, 4, 150, 0.7, 2).unwrap();
        assert!((0.0..=1.0).contains(&acc));
        assert!((0.0..=1.0).contains(&fmt));
        assert!((0.0..=1.0).contains(&passk));
    }
}
