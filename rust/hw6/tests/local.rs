//! Local sanity-check tests for Homework 6.
//!
//! Ported directly from `homework/hw6/hw6_tests.py` (`test_*` functions).
//! Run with: `cargo test -p hw6 --test local`
//!
//! Note: `test_eval_llm_chat` and `test_eval_llm_dpo` download the pretrained checkpoints
//! (`model_base.pth`, `model_chat.pth`) and UltraChat datasets from HuggingFace Hub
//! (`zkolter/Chat-Tuning-Homework`), so they are marked `#[ignore]` to keep the default
//! test suite fast and 100% offline.
//! Run them explicitly with:
//! `cargo test --release -p hw6 --test local test_eval_llm_chat -- --ignored`
//! `cargo test --release -p hw6 --test local test_eval_llm_dpo -- --ignored`

mod local {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use candle_core::{D, DType, Device, Result, Tensor, TensorId, Var, backprop::GradStore};
    use common::{assert_allclose, seeded_randn};
    use hw6::{
        ASSISTANT_CLOSE, ASSISTANT_CLOSE_ID, ASSISTANT_OPEN, Adam, CausalLM, ChatTokenizer,
        DataLoaderChat, Embedding, LLM, Linear, MLP, Message, ModelConfig,
        MultiHeadAttentionKVCache, Optimizer, Tokenizer, TransformerBlock, USER_CLOSE, USER_OPEN,
        convert_to_chat_format, cross_entropy_loss, download_hw6_files, dpo_loss,
        generate_with_writer, get_loss_mask, load_pretrained_llm, log_probs, mask_indices,
        pretokenize_chat, rms_norm, self_attention, silu, softplus, train_dpo, train_llm_chat,
    };

    static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(prefix: &str) -> Self {
            let id = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "hw6_local_{prefix}_{}_{id}",
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

    struct ToyGenerationTokenizer {
        eot: u32,
        vocab: HashMap<u32, &'static str>,
    }

    impl ToyGenerationTokenizer {
        fn new(eot: u32) -> Self {
            let mut vocab = HashMap::new();
            vocab.insert(3, "A");
            vocab.insert(4, "!");
            vocab.insert(5, "B");
            Self { eot, vocab }
        }
    }

    impl Tokenizer for ToyGenerationTokenizer {
        fn encode(&self, _text: &str) -> Vec<u32> {
            Vec::new()
        }

        fn decode(&self, tokens: &[u32]) -> Result<String> {
            let mut s = String::new();
            for t in tokens {
                let piece = self
                    .vocab
                    .get(t)
                    .ok_or_else(|| candle_core::Error::Msg(format!("unknown token {t}")))?;
                s.push_str(piece);
            }
            Ok(s)
        }

        fn special_token(&self, _name: &str) -> Option<u32> {
            None
        }

        fn eot_token(&self) -> u32 {
            self.eot
        }
    }

    struct ToyGenerationModel {
        next_tokens: Vec<u32>,
        vocab_size: usize,
        calls: Vec<(Vec<Vec<u32>>, usize, bool, DType)>,
    }

    impl ToyGenerationModel {
        fn new(next_tokens: Vec<u32>, vocab_size: usize) -> Self {
            Self {
                next_tokens,
                vocab_size,
                calls: Vec::new(),
            }
        }
    }

    impl CausalLM for ToyGenerationModel {
        fn forward(
            &mut self,
            tokens: &Tensor,
            seq_pos: usize,
            use_kv_cache: bool,
        ) -> Result<Tensor> {
            let rows = tokens.to_vec2::<u32>()?;
            let seq_len = tokens.dim(1)?;
            self.calls
                .push((rows, seq_pos, use_kv_cache, tokens.dtype()));
            let mut logits = vec![-1e9_f32; seq_len * self.vocab_size];
            let next_token = self.next_tokens[self.calls.len() - 1] as usize;
            logits[(seq_len - 1) * self.vocab_size + next_token] = 0.0;
            Tensor::from_vec(logits, (1, seq_len, self.vocab_size), tokens.device())
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

    struct FakeChatTokenizer;

    impl Tokenizer for FakeChatTokenizer {
        fn encode(&self, _text: &str) -> Vec<u32> {
            Vec::new()
        }

        fn decode(&self, _tokens: &[u32]) -> Result<String> {
            Ok(String::new())
        }

        fn special_token(&self, name: &str) -> Option<u32> {
            match name {
                USER_OPEN => Some(91),
                USER_CLOSE => Some(92),
                ASSISTANT_OPEN => Some(93),
                ASSISTANT_CLOSE => Some(94),
                _ => None,
            }
        }
    }

    struct TinyChatTrainModel {
        embed: Embedding,
        proj: Linear,
    }

    impl TinyChatTrainModel {
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

    impl CausalLM for TinyChatTrainModel {
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

        fn clone_fresh(&self) -> Self {
            let copied = self.table.as_detached_tensor().copy().unwrap();
            Self {
                table: Var::from_tensor(&copied).unwrap(),
            }
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
        let std = var.sqrt();
        let expected_std = (2.0_f64 / 100.0).sqrt();
        assert!(
            (std - expected_std).abs() < 5e-3,
            "expected Linear weight std ~= {expected_std:.5}, got {std:.5}"
        );
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
        let ref1d = Tensor::from_vec(ref1d_data, (13, 20), &device).unwrap();
        assert_allclose(&out1d, &ref1d, 1e-6, 1e-6);

        let idx2d: Vec<u32> = (0..77).map(|i| (i * 29 + 5) % 200).collect();
        let y2d = Tensor::from_slice(&idx2d, (7, 11), &device).unwrap();
        let out2d = layer.forward(&y2d).unwrap();
        assert_eq!(out2d.dims(), &[7, 11, 20]);
        let ref2d_data: Vec<f32> = idx2d
            .iter()
            .flat_map(|&i| w_rows[i as usize].iter().copied())
            .collect();
        let ref2d = Tensor::from_vec(ref2d_data, (7, 11, 20), &device).unwrap();
        assert_allclose(&out2d, &ref2d, 1e-6, 1e-6);

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
        let std = var.sqrt();
        assert!(
            (std - 1.0).abs() < 3e-2,
            "expected Embedding weight std ~= 1.0, got {std:.5}"
        );
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
    fn test_multi_head_attention_kv_cache() {
        let device = Device::Cpu;
        let mut attn = MultiHeadAttentionKVCache::new(12, 3, 8, &device).unwrap();
        assert_eq!(attn.k_cache.dims(), &[1, 8, 12]);
        assert_eq!(attn.v_cache.dims(), &[1, 8, 12]);

        let wq = seeded_randn(&[12, 12], 50, &device).unwrap();
        let wk = seeded_randn(&[12, 12], 51, &device).unwrap();
        let wv = seeded_randn(&[12, 12], 52, &device).unwrap();
        let wp = seeded_randn(&[12, 12], 53, &device).unwrap();
        attn.wq.weight.set(&wq).unwrap();
        attn.wk.weight.set(&wk).unwrap();
        attn.wv.weight.set(&wv).unwrap();
        attn.wp.weight.set(&wp).unwrap();

        let x = seeded_randn(&[1, 5, 12], 54, &device).unwrap();
        let mask = causal_mask(5, &device);

        let ref_out = ref_mha(&x, &wq, &wk, &wv, &wp, 3, Some(&mask));
        let out = attn.forward(&x, Some(&mask), 0, false).unwrap();
        assert_eq!(out.dims(), &[1, 5, 12]);
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
            &attn.k_cache.narrow(1, 0, 5).unwrap(),
            &k_expected,
            1e-5,
            1e-5,
        );
        assert_allclose(
            &attn.v_cache.narrow(1, 0, 5).unwrap(),
            &v_expected,
            1e-5,
            1e-5,
        );
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
        let y_vec: Vec<u32> = (0..100).map(|i| (i * 13 + 7) % 50).collect();
        let y = Tensor::from_vec(y_vec, (100,), &device).unwrap();
        let mine = cross_entropy_loss(&logits, &y).unwrap();
        assert_eq!(mine.rank(), 0);
        let ref_val = ref_cross_entropy(&logits, &y);
        assert!(
            (mine.to_scalar::<f32>().unwrap() - ref_val).abs() < 1e-5,
            "got {}, expected {ref_val}",
            mine.to_scalar::<f32>().unwrap()
        );

        let logits3 = seeded_randn(&[4, 5, 7], 81, &device).unwrap();
        let y3_vec: Vec<u32> = (0..20).map(|i| (i * 5 + 2) % 7).collect();
        let y3 = Tensor::from_vec(y3_vec, (4, 5), &device).unwrap();
        let mine3 = cross_entropy_loss(&logits3, &y3).unwrap();
        let ref3 = ref_cross_entropy(&logits3, &y3);
        assert!((mine3.to_scalar::<f32>().unwrap() - ref3).abs() < 1e-5);
    }

    #[test]
    fn test_adam() {
        let device = Device::Cpu;
        let l1 = Linear::new(6, 4, &device).unwrap();
        let l2 = Linear::new(4, 3, &device).unwrap();
        l1.weight
            .set(&seeded_randn(&[4, 6], 90, &device).unwrap())
            .unwrap();
        l2.weight
            .set(&seeded_randn(&[3, 4], 91, &device).unwrap())
            .unwrap();

        let r1 = Linear::new(6, 4, &device).unwrap();
        let r2 = Linear::new(4, 3, &device).unwrap();
        r1.weight.set(l1.weight.as_tensor()).unwrap();
        r2.weight.set(l2.weight.as_tensor()).unwrap();

        let mut opt = Adam::new(
            vec![l1.weight.clone(), l2.weight.clone()],
            1e-3,
            (0.9, 0.95),
            1e-8,
        )
        .unwrap();
        let mut ref_opt = RefAdam::new(
            vec![r1.weight.clone(), r2.weight.clone()],
            1e-3,
            (0.9, 0.95),
            1e-8,
        );

        for step in 0..5u64 {
            let x = seeded_randn(&[16, 6], 100 + step, &device).unwrap();
            let y_vec: Vec<u32> = (0..16)
                .map(|i| (((i + step as usize) * 5) % 3) as u32)
                .collect();
            let y = Tensor::from_vec(y_vec, (16,), &device).unwrap();

            opt.zero_grad();
            let out = l2.forward(&l1.forward(&x).unwrap().tanh().unwrap()).unwrap();
            let loss = cross_entropy_loss(&out, &y).unwrap();
            let grads = loss.backward().unwrap();
            opt.step(&grads).unwrap();

            let ref_out = r2.forward(&r1.forward(&x).unwrap().tanh().unwrap()).unwrap();
            let ref_loss = cross_entropy_loss(&ref_out, &y).unwrap();
            let ref_grads = ref_loss.backward().unwrap();
            ref_opt.step(&ref_grads);
        }

        assert_allclose(l1.weight.as_tensor(), r1.weight.as_tensor(), 1e-6, 1e-6);
        assert_allclose(l2.weight.as_tensor(), r2.weight.as_tensor(), 1e-6, 1e-6);
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

        let mask_vals = model.mask.to_vec2::<f32>().unwrap();
        assert_eq!(mask_vals[0][0], 0.0);
        assert!(mask_vals[0][1].is_infinite() && mask_vals[0][1].is_sign_negative());
        assert_eq!(mask_vals[1][0], 0.0);

        let tokens = Tensor::from_slice(&[0u32, 1, 2, 3, 4], (1, 5), &device).unwrap();
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
    fn test_generate() {
        let mut model = ToyGenerationModel::new(vec![3, 4], 6);
        let tokenizer = ToyGenerationTokenizer::new(4);
        let mut buf: Vec<u8> = Vec::new();
        let generated = generate_with_writer(
            &mut model,
            &[1, 2],
            &tokenizer,
            Some(4),
            0.7,
            5,
            Some(&mut buf),
        )
        .unwrap();

        assert_eq!(generated, vec![3, 4]);
        assert_eq!(String::from_utf8(buf).unwrap(), "A!");
        assert_eq!(model.calls.len(), 2);
        assert_eq!(model.calls[0].0, vec![vec![1, 2]]);
        assert_eq!(model.calls[0].1, 0);
        assert!(model.calls[0].2);
        assert_eq!(model.calls[0].3, DType::U32);
        assert_eq!(model.calls[1].0, vec![vec![3]]);
        assert_eq!(model.calls[1].1, 2);
        assert!(model.calls[1].2);
    }

    // ---------------- Part I: Chat SFT ----------------

    #[test]
    fn test_convert_to_chat_format() {
        let messages = vec![
            Message::user("Hello"),
            Message::assistant("Hi!"),
            Message::user("Bye"),
        ];
        let expected = "<USER>Hello</USER><ASSISTANT>Hi!</ASSISTANT><USER>Bye</USER>";
        assert_eq!(convert_to_chat_format(&messages), expected);
    }

    #[test]
    fn test_pretokenize_chat() {
        let chats = vec![
            vec![Message::user("One"), Message::assistant("Two")],
            vec![Message::user("Three")],
        ];
        let text0 = "<USER>One</USER><ASSISTANT>Two</ASSISTANT>";
        let text1 = "<USER>Three</USER>";
        let tokenizer = FixedEncodeTokenizer::new(&[(text0, &[11, 12, 13]), (text1, &[21, 22])]);

        let tmp = TempDir::new("pretokenize");
        let in_path = tmp.join("chats.json");
        let out_path = tmp.join("tokens.json");
        fs::write(&in_path, serde_json::to_string(&chats).unwrap()).unwrap();

        pretokenize_chat(&tokenizer, &in_path, &out_path).unwrap();

        let tokens: Vec<Vec<u32>> =
            serde_json::from_str(&fs::read_to_string(&out_path).unwrap()).unwrap();
        assert_eq!(
            *tokenizer.calls.borrow(),
            vec![text0.to_string(), text1.to_string()]
        );
        assert_eq!(tokens, vec![vec![11, 12, 13], vec![21, 22]]);
    }

    #[test]
    fn test_get_loss_mask() {
        let tokenizer = FakeChatTokenizer;
        let tokens = [7, 93, 1, 2, 94, 8];
        let expected = vec![false, false, true, true, true, false];
        assert_eq!(get_loss_mask(&tokens, &tokenizer), expected);
    }

    #[test]
    fn test_dataloader_chat() {
        let tokenizer = FakeChatTokenizer;
        let chats: Vec<Vec<u32>> = vec![
            vec![1, 93, 10, 11, 94],
            vec![2, 93, 12, 94],
            vec![3, 93, 13, 14, 15, 94],
            vec![4, 93, 16, 94],
            vec![5, 93, 17, 94],
        ];

        let tmp = TempDir::new("dl_chat");
        let path = tmp.join("chat_tokens.json");
        fs::write(&path, serde_json::to_string(&chats).unwrap()).unwrap();

        let loader = DataLoaderChat::new(&path, 6, 2, &tokenizer, Device::Cpu).unwrap();
        assert_eq!(loader.len(), 2);
        let batches: Vec<(Tensor, Tensor, Tensor)> = loader
            .iter()
            .map(|b| b.unwrap())
            .collect();
        assert_eq!(batches.len(), 2);

        let (x0, y0, m0) = &batches[0];
        assert_eq!(
            x0.to_vec2::<u32>().unwrap(),
            vec![vec![1, 93, 10, 11, 94, 0], vec![2, 93, 12, 94, 0, 0]]
        );
        assert_eq!(
            y0.to_vec2::<u32>().unwrap(),
            vec![vec![93, 10, 11, 94, 0, 0], vec![93, 12, 94, 0, 0, 0]]
        );
        assert_eq!(
            m0.to_vec2::<u8>().unwrap(),
            vec![vec![0, 1, 1, 1, 0, 0], vec![0, 1, 1, 0, 0, 0]]
        );

        let (x1, y1, m1) = &batches[1];
        assert_eq!(
            x1.to_vec2::<u32>().unwrap(),
            vec![vec![3, 93, 13, 14, 15, 94], vec![4, 93, 16, 94, 0, 0]]
        );
        assert_eq!(
            y1.to_vec2::<u32>().unwrap(),
            vec![vec![93, 13, 14, 15, 94, 0], vec![93, 16, 94, 0, 0, 0]]
        );
        assert_eq!(
            m1.to_vec2::<u8>().unwrap(),
            vec![vec![0, 1, 1, 1, 1, 0], vec![0, 1, 1, 0, 0, 0]]
        );

        // Second pass must yield identical batches.
        let batches2: Vec<(Tensor, Tensor, Tensor)> = (&loader)
            .into_iter()
            .map(|b| b.unwrap())
            .collect();
        assert_eq!(batches2.len(), 2);
        for ((xa, ya, ma), (xb, yb, mb)) in batches.iter().zip(batches2.iter()) {
            assert_eq!(xa.to_vec2::<u32>().unwrap(), xb.to_vec2::<u32>().unwrap());
            assert_eq!(ya.to_vec2::<u32>().unwrap(), yb.to_vec2::<u32>().unwrap());
            assert_eq!(ma.to_vec2::<u8>().unwrap(), mb.to_vec2::<u8>().unwrap());
        }
    }

    #[test]
    fn test_train_llm_chat() {
        let make_batches = || -> Vec<Result<(Tensor, Tensor, Tensor)>> {
            vec![
                Ok((
                    u32_2d(&[&[0, 1, 2], &[1, 2, 3]]),
                    u32_2d(&[&[1, 2, 3], &[2, 3, 4]]),
                    mask_2d(&[&[false, true, true], &[true, false, true]]),
                )),
                Ok((
                    u32_2d(&[&[2, 3, 4], &[0, 2, 4]]),
                    u32_2d(&[&[3, 4, 0], &[2, 4, 1]]),
                    mask_2d(&[&[true, true, false], &[false, true, true]]),
                )),
            ]
        };

        // Run with max_iter = Some(1) and compare against 1 manual masked-CE SGD step.
        let mut model = TinyChatTrainModel::new_seeded(6, 4, 0);
        let mut ref_model = model.clone_fresh();
        let init_embed = model.embed.weight.as_detached_tensor().copy().unwrap();

        let mut opt = SimpleSgd::new(model.parameters(), 0.1);
        train_llm_chat(&mut model, make_batches(), &mut opt, Some(1)).unwrap();
        assert_eq!(opt.steps, 1);

        // Manual reference step 0.
        let mut ref_opt = SimpleSgd::new(ref_model.parameters(), 0.1);
        let (x0, y0, m0) = make_batches().into_iter().next().unwrap().unwrap();
        let logits0 = ref_model.forward(&x0, 0, false).unwrap();
        let idx0 = mask_indices(&m0).unwrap();
        let loss0 = cross_entropy_loss(
            &logits0.reshape((6, 6)).unwrap().index_select(&idx0, 0).unwrap(),
            &y0.reshape((6,)).unwrap().index_select(&idx0, 0).unwrap(),
        )
        .unwrap();
        ref_opt.step(&loss0.backward().unwrap()).unwrap();

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
        let diff = (model.embed.weight.as_tensor() - &init_embed)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff > 1e-6, "parameters should have updated after 1 step");

        // Run full pass (max_iter = None) and verify both batches were processed.
        let mut model2 = TinyChatTrainModel::new_seeded(6, 4, 0);
        let mut opt2 = SimpleSgd::new(model2.parameters(), 0.1);
        train_llm_chat(&mut model2, make_batches(), &mut opt2, None).unwrap();
        assert_eq!(opt2.steps, 2);
    }

    // ---------------- Part II: Direct Preference Optimization ----------------

    #[test]
    fn test_log_probs() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(
            vec![
                2.0_f32, 0.0, -1.0, 0.5, 1.5, -0.5, 1.0, -1.0, 0.0, // batch 0
                -0.5, 1.0, 0.0, 2.0, 0.0, -2.0, 0.25, 0.25, 0.25, // batch 1
            ],
            (2, 3, 3),
            &device,
        )
        .unwrap();
        let y = u32_2d(&[&[0, 1, 2], &[1, 0, 2]]);
        let mask = mask_2d(&[&[true, false, true], &[true, true, false]]);

        let out = log_probs(&logits, &y, &mask).unwrap();
        assert_eq!(out.dims(), &[2]);
        let expected = Tensor::from_slice(&[-1.5775_f32, -0.6073], (2,), &device).unwrap();
        assert_allclose(&out, &expected, 1e-4, 1e-4);
    }

    #[test]
    fn test_softplus() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[-3.0_f32, -0.5, 0.0, 2.0], (4,), &device).unwrap();
        let out = softplus(&x, 0.7).unwrap();
        let expected = Tensor::from_slice(
            &[0.1155_f32, 0.5334, std::f32::consts::LN_2, 1.6204],
            (4,),
            &device,
        )
        .unwrap();
        assert_allclose(&out, &expected, 1e-4, 1e-4);

        let x2 = Tensor::from_vec(vec![1.0_f32, -1.0, 0.25, -0.25], (2, 2), &device).unwrap();
        let out2 = softplus(&x2, 0.3).unwrap();
        let ref2_vals: Vec<f32> = [1.0_f64, -1.0, 0.25, -0.25]
            .iter()
            .map(|&v| (1.0 + (0.3 * v).exp()).ln() as f32)
            .collect();
        let ref2 = Tensor::from_vec(ref2_vals, (2, 2), &device).unwrap();
        assert_allclose(&out2, &ref2, 1e-6, 1e-6);
    }

    #[test]
    fn test_dpo_loss() {
        let mut model = LookupLM::new(
            vec![
                2.0, 0.0, -1.0, // 0
                0.0, 2.0, -0.5, // 1
                1.0, -0.5, 0.5, // 2
                -1.0, 0.5, 1.5, // 3
            ],
            4,
            3,
        );
        let mut model_ref = LookupLM::new(
            vec![
                1.0, 0.0, -0.5, // 0
                0.0, 1.0, 0.0, // 1
                0.5, -0.5, 0.25, // 2
                -0.5, 0.0, 1.0, // 3
            ],
            4,
            3,
        );

        let xp = u32_2d(&[&[0, 1, 2], &[1, 2, 3]]);
        let yp = u32_2d(&[&[1, 2, 0], &[2, 0, 1]]);
        let maskp = mask_2d(&[&[true, true, false], &[false, true, true]]);
        let xn = u32_2d(&[&[0, 2, 3], &[3, 1, 0]]);
        let yn = u32_2d(&[&[2, 0, 1], &[1, 0, 2]]);
        let maskn = mask_2d(&[&[true, false, true], &[true, true, false]]);

        let loss = dpo_loss(
            &mut model,
            &mut model_ref,
            &xp,
            &yp,
            &maskp,
            &xn,
            &yn,
            &maskn,
            0.3,
        )
        .unwrap();
        assert_eq!(loss.dims(), &[2]);
        let expected = Tensor::from_slice(&[0.8100_f32, 0.5797], (2,), &Device::Cpu).unwrap();
        assert_allclose(&loss, &expected, 1e-4, 1e-4);

        let grads = loss.sum_all().unwrap().backward().unwrap();
        assert!(
            grads.get(model.table.as_tensor()).is_some(),
            "policy model must receive gradients"
        );
        assert!(
            grads.get(model_ref.table.as_tensor()).is_none(),
            "reference model must NOT receive gradients (did you forget `.detach()`?)"
        );
    }

    #[test]
    fn test_train_dpo() {
        let make_pos = || -> Vec<Result<(Tensor, Tensor, Tensor)>> {
            vec![
                Ok((
                    u32_2d(&[&[0, 1]]),
                    u32_2d(&[&[1, 2]]),
                    mask_2d(&[&[true, false]]),
                )),
                Ok((
                    u32_2d(&[&[2, 3]]),
                    u32_2d(&[&[0, 1]]),
                    mask_2d(&[&[false, true]]),
                )),
            ]
        };
        let make_neg = || -> Vec<Result<(Tensor, Tensor, Tensor)>> {
            vec![
                Ok((
                    u32_2d(&[&[3, 2]]),
                    u32_2d(&[&[2, 0]]),
                    mask_2d(&[&[true, true]]),
                )),
                Ok((
                    u32_2d(&[&[1, 0]]),
                    u32_2d(&[&[0, 2]]),
                    mask_2d(&[&[true, false]]),
                )),
            ]
        };

        let init_table = vec![
            1.0_f32, 0.0, -0.5, // 0
            0.0, 1.0, 0.25, // 1
            0.5, -0.5, 0.0, // 2
            -0.25, 0.5, 0.75, // 3
        ];

        // 1 step (max_iter = Some(1))
        let mut model = LookupLM::new(init_table.clone(), 4, 3);
        let mut model_ref = model.clone_fresh();
        let ref_before = model_ref.table.as_detached_tensor().copy().unwrap();
        let mut opt = SimpleSgd::new(vec![model.table.clone()], 0.2);

        train_dpo(
            &mut model,
            &mut model_ref,
            make_pos(),
            make_neg(),
            &mut opt,
            0.2,
            Some(1),
        )
        .unwrap();
        assert_eq!(opt.steps, 1);
        assert_allclose(model_ref.table.as_tensor(), &ref_before, 0.0, 0.0);
        let diff = (model.table.as_tensor() - &ref_before)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff > 1e-5, "policy weights must change during DPO training");

        // Full pass (max_iter = None)
        let mut model2 = LookupLM::new(init_table, 4, 3);
        let mut model_ref2 = model2.clone_fresh();
        let mut opt2 = SimpleSgd::new(vec![model2.table.clone()], 0.2);
        train_dpo(
            &mut model2,
            &mut model_ref2,
            make_pos(),
            make_neg(),
            &mut opt2,
            0.2,
            None,
        )
        .unwrap();
        assert_eq!(opt2.steps, 2);
        assert_allclose(model_ref2.table.as_tensor(), &ref_before, 0.0, 0.0);
    }

    // ---------------- Optional Full HF Checkpoint Evaluations (Offline-Safe) ----------------

    fn masked_nll_and_close_prob(
        model: &mut LLM,
        batches: &[(Tensor, Tensor, Tensor)],
        close_token: u32,
    ) -> (f64, f64) {
        let mut total_loss = 0.0_f64;
        let mut total_tokens = 0usize;
        let mut close_probs = Vec::new();

        for (x, y, mask) in batches {
            let logits = model
                .forward(x, 0, false)
                .unwrap()
                .detach()
                .to_dtype(DType::F32)
                .unwrap();
            let (b, s, v) = logits.dims3().unwrap();
            let idx = mask_indices(mask).unwrap();
            let flat_logits = logits
                .reshape((b * s, v))
                .unwrap()
                .index_select(&idx, 0)
                .unwrap();
            let flat_y = y
                .reshape((b * s,))
                .unwrap()
                .index_select(&idx, 0)
                .unwrap();
            let n = flat_y.elem_count();
            let loss = cross_entropy_loss(&flat_logits, &flat_y)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap() as f64;
            total_loss += loss * n as f64;
            total_tokens += n;

            let y_vec = flat_y.to_vec1::<u32>().unwrap();
            let logits_rows = flat_logits.to_vec2::<f32>().unwrap();
            for (row, &tok) in logits_rows.iter().zip(y_vec.iter()) {
                if tok == close_token {
                    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                    let sum_exp: f64 = row.iter().map(|&z| ((z as f64) - max).exp()).sum();
                    let p = ((row[close_token as usize] as f64) - max).exp() / sum_exp;
                    close_probs.push(p);
                }
            }
        }

        assert!(total_tokens > 0);
        assert!(!close_probs.is_empty());
        let avg_close = close_probs.iter().sum::<f64>() / close_probs.len() as f64;
        (total_loss / total_tokens as f64, avg_close)
    }

    #[test]
    #[ignore = "downloads model_base.pth and ultrachat_short.json from HuggingFace Hub and runs 10 SFT steps (~1-2 min in --release)"]
    fn test_eval_llm_chat() {
        let device = Device::Cpu;
        let files = download_hw6_files().unwrap();
        let tokenizer = ChatTokenizer::new().unwrap();
        let cfg = ModelConfig::from_file(&files.params).unwrap();

        // Pretokenize the first 32 conversations (covers 10 train batches + holdout batch 13).
        let raw_chats: Vec<Vec<Message>> =
            serde_json::from_str(&fs::read_to_string(&files.ultrachat_short).unwrap()).unwrap();
        let tmp = TempDir::new("eval_chat");
        let sub_in = tmp.join("sub.json");
        let sub_tok = tmp.join("sub_tok.json");
        fs::write(&sub_in, serde_json::to_string(&raw_chats[..32]).unwrap()).unwrap();
        pretokenize_chat(&tokenizer, &sub_in, &sub_tok).unwrap();

        let loader = DataLoaderChat::new(&sub_tok, cfg.max_seq_len, 2, &tokenizer, device.clone())
            .unwrap();
        let holdout: Vec<(Tensor, Tensor, Tensor)> =
            loader.iter().skip(13).take(1).map(|b| b.unwrap()).collect();

        let mut model = load_pretrained_llm(&files.params, &files.model_base, &device).unwrap();
        let mut base_model = model.deep_clone().unwrap();
        let mut opt = Adam::new(model.parameters(), 2e-5, (0.9, 0.95), 1e-8).unwrap();
        train_llm_chat(&mut model, &loader, &mut opt, Some(10)).unwrap();

        let (loss, close_prob) = masked_nll_and_close_prob(&mut model, &holdout, ASSISTANT_CLOSE_ID);
        let (base_loss, base_close_prob) =
            masked_nll_and_close_prob(&mut base_model, &holdout, ASSISTANT_CLOSE_ID);
        assert!(loss < base_loss, "loss {loss} should be < base_loss {base_loss}");
        assert!(loss < 2.2, "loss {loss} should be < 2.2");
        assert!(
            close_prob > (0.005_f64).max(1000.0 * base_close_prob),
            "close_prob {close_prob} too low vs base {base_close_prob}"
        );
    }

    #[test]
    #[ignore = "downloads model_chat.pth and UltraChat DPO datasets from HuggingFace Hub and runs 10 DPO steps (~2-4 min in --release)"]
    fn test_eval_llm_dpo() {
        let device = Device::Cpu;
        let files = download_hw6_files().unwrap();
        let tokenizer = ChatTokenizer::new().unwrap();
        let cfg = ModelConfig::from_file(&files.params).unwrap();

        let pos_chats: Vec<Vec<Message>> =
            serde_json::from_str(&fs::read_to_string(&files.ultrachat_dpo_pos).unwrap()).unwrap();
        let neg_chats: Vec<Vec<Message>> =
            serde_json::from_str(&fs::read_to_string(&files.ultrachat_dpo_neg).unwrap()).unwrap();
        let tmp = TempDir::new("eval_dpo");
        let pos_in = tmp.join("pos.json");
        let neg_in = tmp.join("neg.json");
        let pos_tok = tmp.join("pos_tok.json");
        let neg_tok = tmp.join("neg_tok.json");
        fs::write(&pos_in, serde_json::to_string(&pos_chats[..30]).unwrap()).unwrap();
        fs::write(&neg_in, serde_json::to_string(&neg_chats[..30]).unwrap()).unwrap();
        pretokenize_chat(&tokenizer, &pos_in, &pos_tok).unwrap();
        pretokenize_chat(&tokenizer, &neg_in, &neg_tok).unwrap();

        let loader_pos =
            DataLoaderChat::new(&pos_tok, cfg.max_seq_len, 2, &tokenizer, device.clone()).unwrap();
        let loader_neg =
            DataLoaderChat::new(&neg_tok, cfg.max_seq_len, 2, &tokenizer, device.clone()).unwrap();

        let mut model = load_pretrained_llm(&files.params, &files.model_chat, &device).unwrap();
        let mut model_ref = model.deep_clone().unwrap();
        let mut opt = Adam::new(model.parameters(), 1e-6, (0.9, 0.95), 1e-8).unwrap();
        train_dpo(
            &mut model,
            &mut model_ref,
            &loader_pos,
            &loader_neg,
            &mut opt,
            0.1,
            Some(10),
        )
        .unwrap();

        let (xp, yp, maskp) = loader_pos.iter().nth(11).unwrap().unwrap();
        let (xn, yn, maskn) = loader_neg.iter().nth(11).unwrap().unwrap();
        let heldout_loss = dpo_loss(
            &mut model,
            &mut model_ref,
            &xp,
            &yp,
            &maskp,
            &xn,
            &yn,
            &maskn,
            0.1,
        )
        .unwrap()
        .mean_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
        assert!(
            heldout_loss < 0.55,
            "expected held-out DPO loss < 0.55, got {heldout_loss}"
        );
    }
}
