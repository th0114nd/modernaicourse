//! Local sanity-check tests for Homework 5.
//!
//! Ported directly from `homework/hw5/hw5_tests.py` (`test_*` functions).
//! Run with: `cargo test -p hw5 --test local`
//!
//! Note: `test_eval_llm_tinystories` downloads `TinyStoriesV2-GPT4-train.txt` (~2 GB)
//! and trains the 4-layer TinyStories LLM, so it is marked `#[ignore]` to keep the
//! default test suite fast and completely offline.
//! Run it explicitly with:
//! `cargo test --release -p hw5 --test local test_eval_llm_tinystories -- --ignored`

mod local {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use candle_core::{D, DType, Device, Result, Tensor, TensorId, Var, backprop::GradStore};
    use common::{assert_allclose, seeded_randn};
    use hw5::{
        Adam, CausalLM, DataLoader, Embedding, Gpt2Tokenizer, LLM, Linear, MLP,
        MultiHeadAttentionKVCache, Optimizer, Tokenizer, TransformerBlock, bpe_decode, bpe_encode,
        cross_entropy_loss, download_tinystories, generate, generate_with_writer, merge_pair,
        most_common_pair, padded_vocab_size, pretokenize_data, rms_norm, self_attention, silu,
        text_to_corpus, train_bpe, train_llm,
    };

    static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(prefix: &str) -> Self {
            let id = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "hw5_local_{prefix}_{}_{id}",
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

    fn write_u16_tokens(path: &std::path::Path, tokens: &[u16]) {
        let mut bytes = Vec::with_capacity(tokens.len() * 2);
        for &t in tokens {
            bytes.extend_from_slice(&t.to_le_bytes());
        }
        fs::write(path, bytes).unwrap();
    }

    fn read_u16_tokens(path: &std::path::Path) -> Vec<u16> {
        let bytes = fs::read(path).unwrap();
        assert_eq!(bytes.len() % 2, 0);
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| u16::from_le_bytes(c))
            .collect()
    }

    fn vecs_of_str(rows: &[&[&str]]) -> Vec<Vec<String>> {
        rows.iter()
            .map(|row| row.iter().map(|s| (*s).to_string()).collect())
            .collect()
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
        let k_t = k.transpose(D::Minus2, D::Minus1).unwrap().contiguous().unwrap();
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
        let pos = model.pos_embeddings.as_tensor().narrow(0, 0, seq_len).unwrap();
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
                    let m = ((self.m[&id].clone() * b1).unwrap() + (g * (1.0 - b1)).unwrap()).unwrap();
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
    }

    impl Optimizer for SimpleSgd {
        fn step(&mut self, grads: &GradStore) -> Result<()> {
            for p in &self.params {
                if let Some(g) = grads.get(p.as_tensor()) {
                    p.set(&(p.as_tensor() - (g * self.lr)?)?)?;
                }
            }
            Ok(())
        }
    }

    struct ChunkTokenizer {
        calls: RefCell<Vec<String>>,
    }

    impl ChunkTokenizer {
        fn new() -> Self {
            Self {
                calls: RefCell::new(Vec::new()),
            }
        }
    }

    impl Tokenizer for ChunkTokenizer {
        fn encode(&self, text: &str) -> Vec<u32> {
            self.calls.borrow_mut().push(text.to_string());
            text.chars().map(|c| c as u32).collect()
        }

        fn decode(&self, tokens: &[u32]) -> Result<String> {
            Ok(tokens
                .iter()
                .filter_map(|&t| char::from_u32(t))
                .collect())
        }

        fn eot_token(&self) -> u32 {
            0
        }

        fn n_vocab(&self) -> usize {
            256
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

        fn eot_token(&self) -> u32 {
            self.eot
        }

        fn n_vocab(&self) -> usize {
            6
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
        fn forward(&mut self, tokens: &Tensor, seq_pos: usize, use_kv_cache: bool) -> Result<Tensor> {
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

    #[derive(Clone)]
    struct TinyTrainModel {
        embed: Embedding,
        proj: Linear,
        output_dtype: DType,
    }

    impl TinyTrainModel {
        fn new_seeded(vocab_size: usize, dim: usize, seed: u64, output_dtype: DType) -> Self {
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
            Self {
                embed,
                proj,
                output_dtype,
            }
        }

        fn clone_fresh(&self) -> Self {
            let device = self.embed.weight.device();
            let (v, d) = self.embed.weight.dims2().unwrap();
            let embed = Embedding::new(v, d, device).unwrap();
            let proj = Linear::new(d, v, device).unwrap();
            embed.weight.set(self.embed.weight.as_tensor()).unwrap();
            proj.weight.set(self.proj.weight.as_tensor()).unwrap();
            Self {
                embed,
                proj,
                output_dtype: self.output_dtype,
            }
        }

        fn parameters(&self) -> Vec<Var> {
            vec![self.embed.weight.clone(), self.proj.weight.clone()]
        }
    }

    impl CausalLM for TinyTrainModel {
        fn forward(&mut self, tokens: &Tensor, _seq_pos: usize, _use_kv_cache: bool) -> Result<Tensor> {
            let h = self.embed.forward(tokens)?;
            self.proj.forward(&h)?.to_dtype(self.output_dtype)
        }
    }

    fn sequence_loss(model: &mut LLM, tokens: &Tensor) -> f32 {
        let seq_len = tokens.dim(1).unwrap();
        let x = tokens.narrow(1, 0, seq_len - 1).unwrap();
        let y = tokens.narrow(1, 1, seq_len - 1).unwrap();
        let logits = model
            .forward(&x, 0, false)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        cross_entropy_loss(&logits, &y)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    // ---------------- Part I: BPE Tokenizer ----------------

    #[test]
    fn test_text_to_corpus() {
        let (corpus, counts) = text_to_corpus("a b b");
        assert_eq!(corpus, vecs_of_str(&[&["a"], &[" ", "b"]]));
        assert_eq!(counts, vec![1, 2]);

        let (corpus, counts) = text_to_corpus("hi there\nthere");
        assert_eq!(
            corpus,
            vecs_of_str(&[
                &["h", "i"],
                &[" ", "t", "h", "e", "r", "e"],
                &["\n", "t", "h", "e", "r", "e"],
            ])
        );
        assert_eq!(counts, vec![1, 1, 1]);
    }

    #[test]
    fn test_most_common_pair() {
        // Tie between ("a", "b") = 3 and ("b", "c") = 3 must break toward first-seen ("a", "b").
        let corpus = vecs_of_str(&[&["a", "b", "a"], &["a", "b"], &["b", "c"]]);
        let counts = vec![2, 1, 3];
        assert_eq!(
            most_common_pair(&corpus, &counts),
            ("a".to_string(), "b".to_string())
        );

        let corpus = vecs_of_str(&[&[" ", "x"], &[" ", "x", "y"], &["x", "y"]]);
        let counts = vec![4, 1, 1];
        assert_eq!(
            most_common_pair(&corpus, &counts),
            (" ".to_string(), "x".to_string())
        );
    }

    #[test]
    fn test_merge_pair() {
        let mut corpus = vecs_of_str(&[&["a", "b", "c"], &["c", "a", "b"]]);
        merge_pair(&mut corpus, ("a", "b"));
        assert_eq!(corpus, vecs_of_str(&[&["ab", "c"], &["c", "ab"]]));

        let mut corpus = vecs_of_str(&[&["x", "y", "z"], &["x", "y"], &["y", "z"]]);
        merge_pair(&mut corpus, ("y", "z"));
        assert_eq!(
            corpus,
            vecs_of_str(&[&["x", "yz"], &["x", "y"], &["yz"]])
        );
    }

    #[test]
    fn test_train_bpe() {
        let (tokens, merges) = train_bpe("aa aa aa", 258);
        assert_eq!(tokens.len(), 258);
        assert_eq!(tokens["a"], 'a' as usize);
        assert_eq!(tokens[" "], ' ' as usize);
        assert_eq!(tokens["aa"], 256);
        assert_eq!(tokens[" aa"], 257);
        assert_eq!(
            merges,
            vec![
                ("a".to_string(), "a".to_string()),
                (" ".to_string(), "aa".to_string()),
            ]
        );
    }

    #[test]
    fn test_bpe_encode() {
        let tokens: HashMap<String, usize> = [
            ("a".to_string(), 0),
            (" ".to_string(), 1),
            ("aa".to_string(), 2),
            (" aa".to_string(), 3),
        ]
        .into_iter()
        .collect();
        let merges = vec![
            ("a".to_string(), "a".to_string()),
            (" ".to_string(), "aa".to_string()),
        ];
        assert_eq!(bpe_encode("aa aa", &merges, &tokens), vec![2, 3]);
        assert_eq!(bpe_encode("aa", &merges, &tokens), vec![2]);
    }

    #[test]
    fn test_bpe_decode() {
        let tokens: HashMap<String, usize> = [
            ("a".to_string(), 0),
            (" ".to_string(), 1),
            ("aa".to_string(), 2),
            (" aa".to_string(), 3),
        ]
        .into_iter()
        .collect();
        assert_eq!(bpe_decode(&[2, 3], &tokens), "aa aa");
        assert_eq!(bpe_decode(&[0, 1, 0], &tokens), "a a");
    }

    // ---------------- Part II: Transformer Architecture ----------------

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

        // Verify Kaiming-style random normal init std ~= sqrt(2 / in_dim).
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

        // Verify unscaled N(0, 1) init.
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
        assert_allclose(&attn.k_cache.narrow(1, 0, 5).unwrap(), &k_expected, 1e-5, 1e-5);
        assert_allclose(&attn.v_cache.narrow(1, 0, 5).unwrap(), &v_expected, 1e-5, 1e-5);
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
        assert_allclose(&out.narrow(1, 3, 2).unwrap(), &tail, 3e-5, 3e-5);
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
        assert_eq!(model.parameters().len(), 2 + 2 * 6 + 1);

        let mask_vals = model.mask.to_vec2::<f32>().unwrap();
        assert_eq!(mask_vals[0][0], 0.0);
        assert!(mask_vals[0][1].is_infinite() && mask_vals[0][1].is_sign_negative());
        assert_eq!(mask_vals[1][0], 0.0);

        let tokens = Tensor::new(&[[0_u32, 1, 2, 3, 4]], &device).unwrap();
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
        assert_allclose(&out.narrow(1, 3, 2).unwrap(), &tail, 6e-5, 6e-5);
    }

    // ---------------- Part III: Training & Generation ----------------

    #[test]
    fn test_cross_entropy_loss() {
        let device = Device::Cpu;
        let logits = seeded_randn(&[100, 50], 110, &device).unwrap();
        let y_vec: Vec<u32> = (0..100).map(|i| (i * 13 + 7) % 50).collect();
        let y = Tensor::from_vec(y_vec, (100,), &device).unwrap();
        let ref_val = ref_cross_entropy(&logits, &y);
        let mine = cross_entropy_loss(&logits, &y).unwrap();
        assert_eq!(mine.rank(), 0);
        assert!(
            (mine.to_scalar::<f32>().unwrap() - ref_val).abs() < 1e-5,
            "2D cross_entropy mismatch"
        );

        let logits3 = seeded_randn(&[4, 5, 7], 111, &device).unwrap();
        let y3_vec: Vec<u32> = (0..20).map(|i| (i * 5 + 2) % 7).collect();
        let y3 = Tensor::from_vec(y3_vec, (4, 5), &device).unwrap();
        let ref_val3 = ref_cross_entropy(&logits3, &y3);
        let mine3 = cross_entropy_loss(&logits3, &y3).unwrap();
        assert_eq!(mine3.rank(), 0);
        assert!(
            (mine3.to_scalar::<f32>().unwrap() - ref_val3).abs() < 1e-5,
            "3D cross_entropy mismatch"
        );
    }

    #[test]
    fn test_pretokenize_data() {
        let tokenizer = ChunkTokenizer::new();
        let dir = TempDir::new("pretokenize");
        let in_path = dir.join("sample.txt");
        let out_path = dir.join("sample.bin");

        fs::write(&in_path, "abcdefg").unwrap();
        pretokenize_data(&tokenizer, &in_path, &out_path, 3, Some(2)).unwrap();

        assert_eq!(
            *tokenizer.calls.borrow(),
            vec!["abc".to_string(), "def".to_string()]
        );
        assert_eq!(read_u16_tokens(&out_path), vec![97, 98, 99, 100, 101, 102]);
    }

    #[test]
    fn test_dataloader() {
        let dir = TempDir::new("dataloader");
        let bin_path = dir.join("tokens.bin");
        let tokens: Vec<u16> = (0..20).collect();
        write_u16_tokens(&bin_path, &tokens);

        let loader = DataLoader::new(&bin_path, 3, 2, Device::Cpu).unwrap();
        assert_eq!(loader.len(), 3);

        let batches: Vec<(Tensor, Tensor)> = loader.iter().map(|r| r.unwrap()).collect();
        assert_eq!(batches.len(), 3);

        let expected = [
            (vec![vec![0_u32, 1, 2], vec![4, 5, 6]], vec![vec![1_u32, 2, 3], vec![5, 6, 7]]),
            (
                vec![vec![6, 7, 8], vec![10, 11, 12]],
                vec![vec![7, 8, 9], vec![11, 12, 13]],
            ),
            (
                vec![vec![12, 13, 14], vec![16, 17, 18]],
                vec![vec![13, 14, 15], vec![17, 18, 19]],
            ),
        ];

        for ((xb, yb), (x_ref, y_ref)) in batches.iter().zip(expected.iter()) {
            assert_eq!(xb.to_vec2::<u32>().unwrap(), *x_ref);
            assert_eq!(yb.to_vec2::<u32>().unwrap(), *y_ref);
        }

        // Second iteration must restart cleanly and yield identical batches.
        let batches2: Vec<(Tensor, Tensor)> = (&loader).into_iter().map(|r| r.unwrap()).collect();
        assert_eq!(batches2.len(), batches.len());
        for ((x1, y1), (x2, y2)) in batches.iter().zip(batches2.iter()) {
            assert_eq!(x1.to_vec2::<u32>().unwrap(), x2.to_vec2::<u32>().unwrap());
            assert_eq!(y1.to_vec2::<u32>().unwrap(), y2.to_vec2::<u32>().unwrap());
        }
    }

    #[test]
    fn test_adam() {
        let device = Device::Cpu;
        let w1 = Var::from_tensor(&seeded_randn(&[4, 6], 120, &device).unwrap()).unwrap();
        let w2 = Var::from_tensor(&seeded_randn(&[3, 4], 121, &device).unwrap()).unwrap();
        let ref_w1 = Var::from_tensor(w1.as_tensor()).unwrap();
        let ref_w2 = Var::from_tensor(w2.as_tensor()).unwrap();

        let mut opt = Adam::new(vec![w1.clone(), w2.clone()], 1e-3, (0.9, 0.95), 1e-8).unwrap();
        let mut ref_opt = RefAdam::new(vec![ref_w1.clone(), ref_w2.clone()], 1e-3, (0.9, 0.95), 1e-8);

        for step_idx in 0..5u64 {
            let x = seeded_randn(&[16, 6], 130 + step_idx, &device).unwrap();
            let y_vec: Vec<u32> = (0..16)
                .map(|i| (((i + step_idx as usize) * 7) % 3) as u32)
                .collect();
            let y = Tensor::from_vec(y_vec, (16,), &device).unwrap();

            let logits = x
                .matmul(&w1.as_tensor().t().unwrap())
                .unwrap()
                .tanh()
                .unwrap()
                .matmul(&w2.as_tensor().t().unwrap())
                .unwrap();
            let loss = cross_entropy_loss(&logits, &y).unwrap();
            opt.zero_grad();
            let grads = loss.backward().unwrap();
            opt.step(&grads).unwrap();

            let ref_logits = x
                .matmul(&ref_w1.as_tensor().t().unwrap())
                .unwrap()
                .tanh()
                .unwrap()
                .matmul(&ref_w2.as_tensor().t().unwrap())
                .unwrap();
            let ref_loss = cross_entropy_loss(&ref_logits, &y).unwrap();
            let ref_grads = ref_loss.backward().unwrap();
            ref_opt.step(&ref_grads);
        }

        assert_allclose(w1.as_tensor(), ref_w1.as_tensor(), 1e-5, 1e-5);
        assert_allclose(w2.as_tensor(), ref_w2.as_tensor(), 1e-5, 1e-5);
        assert_eq!(opt.t, 5);
        assert!(opt.m.contains_key(&w1.id()));
        assert!(opt.v.contains_key(&w2.id()));
    }

    #[test]
    fn test_train_llm() {
        let device = Device::Cpu;
        // Output F64 logits to verify `train_llm` casts logits to DType::F32 before cross-entropy.
        let mut model = TinyTrainModel::new_seeded(5, 4, 140, DType::F64);
        let mut ref_model = model.clone_fresh();

        let mut opt = SimpleSgd {
            params: model.parameters(),
            lr: 0.2,
        };
        let mut ref_opt = SimpleSgd {
            params: ref_model.parameters(),
            lr: 0.2,
        };

        let batches = vec![
            (
                Tensor::new(&[[0_u32, 1, 2], [1, 2, 3]], &device).unwrap(),
                Tensor::new(&[[1_u32, 2, 3], [2, 3, 4]], &device).unwrap(),
            ),
            (
                Tensor::new(&[[2_u32, 3, 4], [0, 2, 4]], &device).unwrap(),
                Tensor::new(&[[3_u32, 4, 0], [2, 4, 1]], &device).unwrap(),
            ),
        ];

        let loader_iter = batches
            .iter()
            .map(|(x, y)| Ok((x.clone(), y.clone())));
        train_llm(&mut model, loader_iter, &mut opt).unwrap();

        for (x, y) in &batches {
            let logits = ref_model
                .forward(x, 0, false)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let loss = cross_entropy_loss(&logits, y).unwrap();
            let grads = loss.backward().unwrap();
            ref_opt.step(&grads).unwrap();
        }

        for (p, p_ref) in model.parameters().iter().zip(ref_model.parameters().iter()) {
            assert_allclose(p.as_tensor(), p_ref.as_tensor(), 1e-5, 1e-5);
        }
    }

    #[test]
    fn test_generate() {
        let mut model = ToyGenerationModel::new(vec![3, 4], 6);
        let tokenizer = ToyGenerationTokenizer::new(4);
        let mut out = Vec::new();

        let generated =
            generate_with_writer(&mut model, &[1, 2], &tokenizer, 0.7, 5, Some(&mut out)).unwrap();

        assert_eq!(generated, vec![3, 4]);
        assert_eq!(String::from_utf8(out).unwrap(), "A!");
        assert_eq!(model.calls.len(), 2);
        assert_eq!(model.calls[0], (vec![vec![1, 2]], 0, true, DType::U32));
        assert_eq!(model.calls[1], (vec![vec![3]], 2, true, DType::U32));

        let mut model2 = ToyGenerationModel::new(vec![3, 4], 6);
        let generated2 = generate(&mut model2, &[1, 2], &tokenizer, 0.7, 5, false).unwrap();
        assert_eq!(generated2, vec![3, 4]);
    }

    #[test]
    fn test_eval_llm_synthetic() {
        // Fast offline end-to-end test: pretokenize -> DataLoader -> Adam -> train_llm -> eval + KV cache.
        let device = Device::Cpu;
        let dir = TempDir::new("eval_synth");
        let txt_path = dir.join("tiny.txt");
        let bin_path = dir.join("tiny.bin");

        // Repeating structured alphabet pattern with vocab < 32 so a tiny LLM learns it in < 1s.
        let pattern = "abcdefghijklmnop";
        let full_text = pattern.repeat(40);
        fs::write(&txt_path, &full_text).unwrap();

        struct AlphaTokenizer;
        impl Tokenizer for AlphaTokenizer {
            fn encode(&self, text: &str) -> Vec<u32> {
                text.bytes().map(|b| (b - b'a') as u32).collect()
            }
            fn decode(&self, tokens: &[u32]) -> Result<String> {
                Ok(tokens.iter().map(|&t| (b'a' + t as u8) as char).collect())
            }
            fn eot_token(&self) -> u32 {
                15
            }
            fn n_vocab(&self) -> usize {
                16
            }
        }

        let tokenizer = AlphaTokenizer;
        pretokenize_data(&tokenizer, &txt_path, &bin_path, 128, None).unwrap();

        let loader = DataLoader::new(&bin_path, 16, 4, device.clone()).unwrap();
        assert!(!loader.is_empty());

        let mut model = LLM::new(16, 16, 4, 24, 32, 2, &device).unwrap();
        let mut opt = Adam::new(model.parameters(), 2e-2, (0.9, 0.95), 1e-8).unwrap();

        for _ in 0..8 {
            train_llm(&mut model, &loader, &mut opt).unwrap();
        }

        let eval_ids: Vec<u32> = (0..16).collect();
        let eval_tokens = Tensor::from_slice(&eval_ids, (1, 16), &device).unwrap();
        let phrase_loss = sequence_loss(&mut model, &eval_tokens);

        let mut corrupted_ids = eval_ids.clone();
        corrupted_ids[1..].reverse();
        let corrupted_tokens = Tensor::from_slice(&corrupted_ids, (1, 16), &device).unwrap();
        let corrupted_loss = sequence_loss(&mut model, &corrupted_tokens);

        assert!(
            phrase_loss < 1.0,
            "expected synthetic trained LLM loss < 1.0, got {phrase_loss:.4}"
        );
        assert!(
            phrase_loss < corrupted_loss,
            "expected phrase_loss ({phrase_loss:.4}) < corrupted_loss ({corrupted_loss:.4})"
        );

        let full = model
            .forward(&eval_tokens.narrow(1, 0, 15).unwrap(), 0, false)
            .unwrap();
        model
            .forward(&eval_tokens.narrow(1, 0, 14).unwrap(), 0, true)
            .unwrap();
        let tail = model
            .forward(&eval_tokens.narrow(1, 14, 1).unwrap(), 14, true)
            .unwrap();
        assert_allclose(&full.narrow(1, 14, 1).unwrap(), &tail, 3e-4, 3e-4);

        // Verify generation runs end-to-end without error.
        let generated =
            generate_with_writer(&mut model, &[0, 1, 2], &tokenizer, 0.5, 8, None).unwrap();
        assert!(!generated.is_empty() && generated.len() <= 8);
    }

    #[test]
    #[ignore = "downloads ~2 GB TinyStories dataset from HuggingFace and trains a 4-layer LLM"]
    fn test_eval_llm_tinystories() {
        let device = Device::Cpu;
        let txt_path = download_tinystories().unwrap();
        let dir = TempDir::new("tinystories");
        let bin_path = dir.join("TinyStoriesV2-GPT4-train.small.bin");

        let tokenizer = Gpt2Tokenizer::new().unwrap();
        pretokenize_data(&tokenizer, &txt_path, &bin_path, 1 << 20, Some(2)).unwrap();

        let loader = DataLoader::new(&bin_path, 512, 16, device.clone()).unwrap();
        let mut model = LLM::new(
            padded_vocab_size(tokenizer.n_vocab()),
            256,
            8,
            512,
            512,
            4,
            &device,
        )
        .unwrap();
        let mut opt = Adam::new(model.parameters(), 1e-3, (0.9, 0.95), 1e-8).unwrap();
        train_llm(&mut model, &loader, &mut opt).unwrap();

        let sample = fs::read_to_string(&txt_path).unwrap();
        let prefix: String = sample.chars().take(4096).collect();
        let all_tokens = tokenizer.encode(&prefix);
        let eval_slice = &all_tokens[0..48];
        let eval_tokens = Tensor::from_slice(eval_slice, (1, 48), &device).unwrap();

        let phrase_loss = sequence_loss(&mut model, &eval_tokens);
        let mut corrupted_vec = eval_slice.to_vec();
        corrupted_vec[1..].reverse();
        let corrupted = Tensor::from_slice(&corrupted_vec, (1, 48), &device).unwrap();
        let corrupted_loss = sequence_loss(&mut model, &corrupted);

        assert!(phrase_loss < 7.0, "expected phrase_loss < 7.0, got {phrase_loss}");
        assert!(phrase_loss < corrupted_loss);

        let full = model
            .forward(&eval_tokens.narrow(1, 0, 47).unwrap(), 0, false)
            .unwrap();
        model
            .forward(&eval_tokens.narrow(1, 0, 46).unwrap(), 0, true)
            .unwrap();
        let tail = model
            .forward(&eval_tokens.narrow(1, 46, 1).unwrap(), 46, true)
            .unwrap();
        assert_allclose(&full.narrow(1, 46, 1).unwrap(), &tail, 3e-4, 3e-4);
    }
}
