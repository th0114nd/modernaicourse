//! Grader-equivalent submission tests for Homework 5.
//!
//! Ported from `homework/hw5/hw5_tests.py` (`submit_*` functions) with verified
//! ground-truth assertions plus extra edge cases.
//! Run with: `cargo test -p hw5 --test submit`

mod submit {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use candle_core::{D, DType, Device, Result, Tensor, Var, backprop::GradStore};
    use common::{assert_allclose, seeded_randn};
    use hw5::{
        Adam, CausalLM, DataLoader, Embedding, Gpt2Tokenizer, LLM, Linear, MLP,
        MultiHeadAttentionKVCache, Optimizer, Tokenizer, TransformerBlock, bpe_decode, bpe_encode,
        cross_entropy_loss, generate_with_writer, merge_pair, most_common_pair, padded_vocab_size,
        pretokenize_data, rms_norm, self_attention, silu, text_to_corpus, train_bpe, train_llm,
    };

    static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(prefix: &str) -> Self {
            let id = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "hw5_submit_{prefix}_{}_{id}",
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

    fn eye(n: usize, device: &Device) -> Tensor {
        let mut data = vec![0.0_f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Tensor::from_vec(data, (n, n), device).unwrap()
    }

    /// Independent f64 scaled dot-product attention for exact submit verification.
    fn ref_sdpa_f64(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let d = q.dim(D::Minus1).unwrap() as f64;
        let q64 = q.to_dtype(DType::F64).unwrap().contiguous().unwrap();
        let k64 = k
            .to_dtype(DType::F64)
            .unwrap()
            .transpose(D::Minus2, D::Minus1)
            .unwrap()
            .contiguous()
            .unwrap();
        let v64 = v.to_dtype(DType::F64).unwrap().contiguous().unwrap();
        let mut scores = (q64.matmul(&k64).unwrap() / d.sqrt()).unwrap();
        if let Some(m) = mask {
            scores = scores
                .broadcast_add(&m.to_dtype(DType::F64).unwrap())
                .unwrap();
        }
        let max = scores.max_keepdim(D::Minus1).unwrap();
        let exp = scores.broadcast_sub(&max).unwrap().exp().unwrap();
        let sum = exp.sum_keepdim(D::Minus1).unwrap();
        exp.broadcast_div(&sum)
            .unwrap()
            .matmul(&v64)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
    }

    fn ref_rms_norm_f64(x: &Tensor, eps: f64) -> Tensor {
        let x64 = x.to_dtype(DType::F64).unwrap();
        let mean_sq = x64.sqr().unwrap().mean_keepdim(D::Minus1).unwrap();
        x64.broadcast_div(&(mean_sq + eps).unwrap().sqrt().unwrap())
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
    }

    fn ref_silu_f64(x: &Tensor) -> Tensor {
        let x64 = x.to_dtype(DType::F64).unwrap();
        let denom = (x64.neg().unwrap().exp().unwrap() + 1.0).unwrap();
        x64.div(&denom).unwrap().to_dtype(DType::F32).unwrap()
    }

    fn ref_mha_f64(
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
        let out = ref_sdpa_f64(&proj(wq), &proj(wk), &proj(wv), mask)
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((b, s, d))
            .unwrap();
        out.broadcast_matmul(&wp.t().unwrap()).unwrap()
    }

    fn ref_block_f64(block: &TransformerBlock, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let x_norm = ref_rms_norm_f64(x, 1e-5);
        let attn = ref_mha_f64(
            &x_norm,
            block.attn.wq.weight.as_tensor(),
            block.attn.wk.weight.as_tensor(),
            block.attn.wv.weight.as_tensor(),
            block.attn.wp.weight.as_tensor(),
            block.attn.n_heads,
            mask,
        );
        let z = (x + attn).unwrap();
        let z_norm = ref_rms_norm_f64(&z, 1e-5);
        let h = z_norm
            .broadcast_matmul(&block.mlp.w1.weight.as_tensor().t().unwrap())
            .unwrap();
        let mlp_out = ref_silu_f64(&h)
            .broadcast_matmul(&block.mlp.w2.weight.as_tensor().t().unwrap())
            .unwrap();
        (&z + mlp_out).unwrap()
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
            Ok(tokens.iter().filter_map(|&t| char::from_u32(t)).collect())
        }

        fn eot_token(&self) -> u32 {
            0
        }

        fn n_vocab(&self) -> usize {
            65536
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
        fn forward(&mut self, tokens: &Tensor, _seq_pos: usize, _use_kv_cache: bool) -> Result<Tensor> {
            let h = self.embed.forward(tokens)?;
            self.proj.forward(&h)
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
        calls: Vec<(Vec<Vec<u32>>, usize, bool)>,
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
            self.calls.push((rows, seq_pos, use_kv_cache));
            let mut logits = vec![-1e9_f32; seq_len * self.vocab_size];
            let next_token = self.next_tokens[self.calls.len() - 1] as usize;
            logits[(seq_len - 1) * self.vocab_size + next_token] = 0.0;
            Tensor::from_vec(logits, (1, seq_len, self.vocab_size), tokens.device())
        }
    }

    // ---------------- Part I: BPE Tokenizer ----------------

    #[test]
    fn submit_text_to_corpus() {
        let (corpus, counts) = text_to_corpus("x y y z");
        assert_eq!(corpus, vecs_of_str(&[&["x"], &[" ", "y"], &[" ", "z"]]));
        assert_eq!(counts, vec![1, 2, 1]);

        let (corpus, counts) = text_to_corpus("go\nstop\nstop");
        assert_eq!(
            corpus,
            vecs_of_str(&[&["g", "o"], &["\n", "s", "t", "o", "p"]])
        );
        assert_eq!(counts, vec![1, 2]);

        // Extra edge cases: leading whitespace, consecutive whitespace, empty string.
        let (corpus, counts) = text_to_corpus(" test target test");
        assert_eq!(
            corpus,
            vecs_of_str(&[
                &[" ", "t", "e", "s", "t"],
                &[" ", "t", "a", "r", "g", "e", "t"],
            ])
        );
        assert_eq!(counts, vec![2, 1]);

        let (corpus, counts) = text_to_corpus("a  b");
        assert_eq!(corpus, vecs_of_str(&[&["a"], &[" "], &[" ", "b"]]));
        assert_eq!(counts, vec![1, 1, 1]);

        let (corpus, counts) = text_to_corpus("");
        assert!(corpus.is_empty() && counts.is_empty());
    }

    #[test]
    fn submit_most_common_pair() {
        let corpus = vecs_of_str(&[&["a", "a"], &["a", "b", "a"], &["b", "a"]]);
        let counts = vec![1, 2, 3];
        assert_eq!(
            most_common_pair(&corpus, &counts),
            ("b".to_string(), "a".to_string())
        );

        let corpus = vecs_of_str(&[&["x", "y", "z"], &["y", "z"], &["x", "y"]]);
        let counts = vec![1, 4, 2];
        assert_eq!(
            most_common_pair(&corpus, &counts),
            ("y".to_string(), "z".to_string())
        );

        // Edge case: repeated pair inside a single word (`"aaa"` has two ("a", "a") windows).
        let corpus = vecs_of_str(&[&["a", "a", "a"], &["b", "c"]]);
        let counts = vec![2, 3];
        assert_eq!(
            most_common_pair(&corpus, &counts),
            ("a".to_string(), "a".to_string())
        );
    }

    #[test]
    fn submit_merge_pair() {
        let mut corpus = vecs_of_str(&[&["a", "b", "c", "d"], &["a", "b"], &["c", "d"]]);
        merge_pair(&mut corpus, ("a", "b"));
        assert_eq!(
            corpus,
            vecs_of_str(&[&["ab", "c", "d"], &["ab"], &["c", "d"]])
        );

        let mut corpus = vecs_of_str(&[&["x", "y", "z"], &["y", "z", "x"]]);
        merge_pair(&mut corpus, ("y", "z"));
        assert_eq!(corpus, vecs_of_str(&[&["x", "yz"], &["yz", "x"]]));

        // Edge case: overlapping triple merges greedily left-to-right.
        let mut corpus = vecs_of_str(&[&["a", "a", "a"], &["a", "a", "a", "a"]]);
        merge_pair(&mut corpus, ("a", "a"));
        assert_eq!(corpus, vecs_of_str(&[&["aa", "a"], &["aa", "aa"]]));
    }

    #[test]
    fn submit_train_bpe() {
        let (tokens, merges) = train_bpe("aa aa aa", 258);
        assert_eq!(
            merges,
            vec![
                ("a".to_string(), "a".to_string()),
                (" ".to_string(), "aa".to_string()),
            ]
        );
        assert_eq!(tokens["aa"], 256);
        assert_eq!(tokens[" aa"], 257);

        // Verify all 256 base Latin-1 characters are present with ID == byte value.
        for b in 0u8..=255 {
            assert_eq!(tokens[&char::from(b).to_string()], b as usize);
        }
    }

    #[test]
    fn submit_bpe_encode() {
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
        assert_eq!(bpe_encode("aa aa aa", &merges, &tokens), vec![2, 3, 3]);
        assert_eq!(bpe_encode("aa", &merges, &tokens), vec![2]);
    }

    #[test]
    fn submit_bpe_decode() {
        let tokens: HashMap<String, usize> = [
            ("a".to_string(), 0),
            (" ".to_string(), 1),
            ("aa".to_string(), 2),
            (" aa".to_string(), 3),
        ]
        .into_iter()
        .collect();
        assert_eq!(bpe_decode(&[2, 3, 3], &tokens), "aa aa aa");
        assert_eq!(bpe_decode(&[0, 1, 0], &tokens), "a a");

        // Full end-to-end BPE train -> encode -> decode round-trip on multi-line story text.
        let train_text = "Once upon a time, a little cat sat on a mat.\nOnce upon a time, a little dog sat on a log.\n";
        let (learned_tokens, learned_merges) = train_bpe(train_text, 280);
        assert_eq!(learned_tokens.len(), 280);
        assert_eq!(learned_merges.len(), 24);
        let test_text = "Once upon a time, the little cat saw a dog!\n";
        let ids = bpe_encode(test_text, &learned_merges, &learned_tokens);
        assert!(ids.len() < test_text.len());
        assert_eq!(bpe_decode(&ids, &learned_tokens), test_text);
    }

    // ---------------- Part II: Transformer Architecture ----------------

    #[test]
    fn submit_linear() {
        let device = Device::Cpu;
        let layer = Linear::new(4, 3, &device).unwrap();
        layer
            .weight
            .set(
                &Tensor::new(
                    &[
                        [1.0_f32, -1.0, 0.5, 2.0],
                        [-0.5, 0.0, 1.0, -1.5],
                        [2.0, 1.5, -0.5, 0.25],
                    ],
                    &device,
                )
                .unwrap(),
            )
            .unwrap();

        let x = Tensor::new(&[[1.0_f32, 2.0, -1.0, 0.5], [0.0, -1.0, 2.0, 3.0]], &device).unwrap();
        let expected_2d =
            Tensor::new(&[[-0.5_f32, -2.25, 5.625], [8.0, -2.5, -1.75]], &device).unwrap();
        assert_allclose(&layer.forward(&x).unwrap(), &expected_2d, 1e-5, 1e-5);

        let x3d = Tensor::stack(&[&x, &x.neg().unwrap()], 0).unwrap();
        let expected_3d =
            Tensor::stack(&[&expected_2d, &expected_2d.neg().unwrap()], 0).unwrap();
        let out3d = layer.forward(&x3d).unwrap();
        assert_allclose(&out3d, &expected_3d, 1e-5, 1e-5);

        // Verify autograd flows through Linear::forward.
        let grads = out3d.sum_all().unwrap().backward().unwrap();
        assert!(grads.get(layer.weight.as_tensor()).is_some());
    }

    #[test]
    fn submit_embedding() {
        let device = Device::Cpu;
        let layer = Embedding::new(8, 3, &device).unwrap();
        layer
            .weight
            .set(
                &Tensor::new(
                    &[
                        [0.0_f32, 0.5, 1.0],
                        [1.0, -1.0, 0.0],
                        [0.25, 0.5, 0.75],
                        [-0.5, 1.5, -1.0],
                        [2.0, 0.0, 1.0],
                        [1.25, -0.25, 0.5],
                        [0.75, 0.25, -0.75],
                        [-1.0, -0.5, 0.5],
                    ],
                    &device,
                )
                .unwrap(),
            )
            .unwrap();

        let y2d = Tensor::new(&[[0_u32, 3, 5], [6, 1, 4]], &device).unwrap();
        let expected_2d = Tensor::new(
            &[
                [[0.0_f32, 0.5, 1.0], [-0.5, 1.5, -1.0], [1.25, -0.25, 0.5]],
                [[0.75, 0.25, -0.75], [1.0, -1.0, 0.0], [2.0, 0.0, 1.0]],
            ],
            &device,
        )
        .unwrap();
        let out2d = layer.forward(&y2d).unwrap();
        assert_allclose(&out2d, &expected_2d, 1e-6, 1e-6);

        let y1d = Tensor::new(&[7_u32, 2, 0], &device).unwrap();
        let expected_1d = Tensor::new(
            &[[-1.0_f32, -0.5, 0.5], [0.25, 0.5, 0.75], [0.0, 0.5, 1.0]],
            &device,
        )
        .unwrap();
        assert_allclose(&layer.forward(&y1d).unwrap(), &expected_1d, 1e-6, 1e-6);

        let grads = out2d.sum_all().unwrap().backward().unwrap();
        assert!(grads.get(layer.weight.as_tensor()).is_some());
    }

    #[test]
    fn submit_silu() {
        let device = Device::Cpu;
        let x = Tensor::new(
            &[[-2.0_f32, -0.5, 0.0, 1.0], [2.0, 3.0, -1.0, 0.25]],
            &device,
        )
        .unwrap();
        assert_allclose(&silu(&x).unwrap(), &ref_silu_f64(&x), 1e-5, 1e-6);

        let x1 = Tensor::new(&[-3.0_f32, 0.0, 3.0], &device).unwrap();
        assert_allclose(&silu(&x1).unwrap(), &ref_silu_f64(&x1), 1e-5, 1e-6);
    }

    #[test]
    fn submit_rms_norm() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[1.0_f32, -1.0, 0.5, 0.5], [2.0, 0.0, -2.0, 1.0]], &device).unwrap();
        assert_allclose(
            &rms_norm(&x, 1e-4).unwrap(),
            &ref_rms_norm_f64(&x, 1e-4),
            1e-5,
            1e-5,
        );

        let x3 = x.unsqueeze(0).unwrap();
        assert_allclose(
            &rms_norm(&x3, 1e-5).unwrap(),
            &ref_rms_norm_f64(&x3, 1e-5),
            1e-5,
            1e-5,
        );
    }

    #[test]
    fn submit_self_attention() {
        let device = Device::Cpu;
        let q = Tensor::new(&[[1.0_f32, 0.0], [0.0, 1.0], [1.0, 1.0]], &device).unwrap();
        let k = Tensor::new(&[[1.0_f32, 0.5], [0.0, 1.0], [1.0, -1.0]], &device).unwrap();
        let v = Tensor::new(&[[0.5_f32, 1.0], [1.5, -0.5], [-1.0, 0.25]], &device).unwrap();
        let mask = causal_mask(3, &device);

        assert_allclose(
            &self_attention(&q, &k, &v, Some(&mask)).unwrap(),
            &ref_sdpa_f64(&q, &k, &v, Some(&mask)),
            1e-5,
            1e-5,
        );
        assert_allclose(
            &self_attention(&q, &k, &v, None).unwrap(),
            &ref_sdpa_f64(&q, &k, &v, None),
            1e-5,
            1e-5,
        );
    }

    #[test]
    fn submit_multi_head_attention_kv_cache() {
        let device = Device::Cpu;
        let mut attn = MultiHeadAttentionKVCache::new(4, 2, 6, &device).unwrap();
        attn.wq.weight.set(&eye(4, &device)).unwrap();
        attn.wk
            .weight
            .set(
                &Tensor::new(
                    &[
                        [1.0_f32, 0.5, 0.0, 0.0],
                        [0.0, 1.0, 0.5, 0.0],
                        [0.0, 0.0, 1.0, 0.5],
                        [0.5, 0.0, 0.0, 1.0],
                    ],
                    &device,
                )
                .unwrap(),
            )
            .unwrap();
        attn.wv
            .weight
            .set(
                &Tensor::new(
                    &[
                        [1.0_f32, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, -1.0],
                    ],
                    &device,
                )
                .unwrap(),
            )
            .unwrap();
        attn.wp.weight.set(&eye(4, &device)).unwrap();

        let x = Tensor::new(
            &[[
                [1.0_f32, 0.0, -1.0, 0.5],
                [0.0, 1.0, 0.5, -0.5],
                [1.0, 1.0, 0.0, 1.0],
                [-1.0, 0.5, 1.0, 0.0],
            ]],
            &device,
        )
        .unwrap();
        let mask = causal_mask(4, &device);

        let ref_full = ref_mha_f64(
            &x,
            attn.wq.weight.as_tensor(),
            attn.wk.weight.as_tensor(),
            attn.wv.weight.as_tensor(),
            attn.wp.weight.as_tensor(),
            2,
            Some(&mask),
        );
        let full = attn.forward(&x, Some(&mask), 0, false).unwrap();
        let prefix = attn
            .forward(
                &x.narrow(1, 0, 2).unwrap(),
                Some(&mask.narrow(0, 0, 2).unwrap().narrow(1, 0, 2).unwrap()),
                0,
                true,
            )
            .unwrap();
        let tail = attn
            .forward(
                &x.narrow(1, 2, 2).unwrap(),
                Some(&mask.narrow(0, 2, 2).unwrap()),
                2,
                true,
            )
            .unwrap();

        assert_allclose(&full, &ref_full, 1e-5, 1e-5);
        assert_allclose(&prefix, &ref_full.narrow(1, 0, 2).unwrap(), 1e-5, 1e-5);
        assert_allclose(&tail, &ref_full.narrow(1, 2, 2).unwrap(), 1e-5, 1e-5);
        assert_allclose(
            &attn.k_cache.narrow(1, 0, 4).unwrap(),
            &attn.wk.forward(&x).unwrap(),
            1e-5,
            1e-5,
        );
    }

    #[test]
    fn submit_mlp() {
        let device = Device::Cpu;
        let mlp = MLP::new(2, 3, &device).unwrap();
        mlp.w1
            .weight
            .set(&Tensor::new(&[[1.0_f32, 0.0], [0.0, 1.0], [1.0, -1.0]], &device).unwrap())
            .unwrap();
        mlp.w2
            .weight
            .set(&Tensor::new(&[[1.0_f32, 0.0, -1.0], [0.0, 1.0, 1.0]], &device).unwrap())
            .unwrap();

        let x = Tensor::new(&[[1.0_f32, -1.0], [2.0, 0.0], [0.5, 0.5]], &device).unwrap();
        let h = x
            .broadcast_matmul(&mlp.w1.weight.as_tensor().t().unwrap())
            .unwrap();
        let expected = ref_silu_f64(&h)
            .broadcast_matmul(&mlp.w2.weight.as_tensor().t().unwrap())
            .unwrap();
        assert_allclose(&mlp.forward(&x).unwrap(), &expected, 1e-5, 1e-5);
    }

    #[test]
    fn submit_transformer_block() {
        let device = Device::Cpu;
        let mut block = TransformerBlock::new(4, 2, 6, 5, &device).unwrap();
        block.attn.wq.weight.set(&eye(4, &device)).unwrap();
        block.attn.wk.weight.set(&eye(4, &device)).unwrap();
        block.attn.wv.weight.set(&eye(4, &device)).unwrap();
        block.attn.wp.weight.set(&eye(4, &device)).unwrap();
        block
            .mlp
            .w1
            .weight
            .set(
                &Tensor::new(
                    &[
                        [1.0_f32, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [1.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, -1.0],
                    ],
                    &device,
                )
                .unwrap(),
            )
            .unwrap();
        block
            .mlp
            .w2
            .weight
            .set(
                &Tensor::new(
                    &[
                        [1.0_f32, 0.0, 0.0, 0.0, 0.5, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.5],
                        [0.0, 0.0, 1.0, 0.0, -0.5, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, -0.5],
                    ],
                    &device,
                )
                .unwrap(),
            )
            .unwrap();

        let x = Tensor::new(
            &[[
                [1.0_f32, 0.0, -1.0, 0.5],
                [0.0, 1.0, 0.5, -0.5],
                [1.0, 1.0, 0.0, 1.0],
            ]],
            &device,
        )
        .unwrap();
        let mask = causal_mask(3, &device);
        let expected = ref_block_f64(&block, &x, Some(&mask));

        let full = block.forward(&x, Some(&mask), 0, false).unwrap();
        assert_allclose(&full, &expected, 1e-5, 1e-5);

        block
            .forward(
                &x.narrow(1, 0, 2).unwrap(),
                Some(&mask.narrow(0, 0, 2).unwrap().narrow(1, 0, 2).unwrap()),
                0,
                true,
            )
            .unwrap();
        let tail = block
            .forward(
                &x.narrow(1, 2, 1).unwrap(),
                Some(&mask.narrow(0, 2, 1).unwrap()),
                2,
                true,
            )
            .unwrap();
        assert_allclose(&tail, &expected.narrow(1, 2, 1).unwrap(), 1e-5, 1e-5);
    }

    #[test]
    fn submit_llm() {
        let device = Device::Cpu;
        let mut model = LLM::new(7, 4, 2, 6, 6, 1, &device).unwrap();
        let tokens = Tensor::new(&[[2_u32, 3, 4]], &device).unwrap();

        // Check causal mask upper triangle is -inf, lower triangle + diag is 0.0.
        let mask4 = model
            .mask
            .narrow(0, 0, 4)
            .unwrap()
            .narrow(1, 0, 4)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for (r, row) in mask4.iter().enumerate() {
            for (c, &val) in row.iter().enumerate() {
                if c > r {
                    assert!(val.is_infinite() && val.is_sign_negative());
                } else {
                    assert_eq!(val, 0.0);
                }
            }
        }

        // Reference forward pass.
        let x0 = model
            .embedding
            .forward(&tokens)
            .unwrap()
            .broadcast_add(&model.pos_embeddings.as_tensor().narrow(0, 0, 3).unwrap())
            .unwrap();
        let m3 = model
            .mask
            .narrow(0, 0, 3)
            .unwrap()
            .narrow(1, 0, 3)
            .unwrap();
        let h = ref_block_f64(&model.layers[0], &x0, Some(&m3));
        let expected = ref_rms_norm_f64(&h, 1e-5)
            .broadcast_matmul(&model.output.weight.as_tensor().t().unwrap())
            .unwrap();

        let out = model.forward(&tokens, 0, false).unwrap();
        assert_allclose(&out, &expected, 3e-5, 3e-5);

        // Verify gradients flow to every parameter (including pos_embeddings) during training!
        let target = Tensor::new(&[[3_u32, 4, 5]], &device).unwrap();
        let loss = cross_entropy_loss(&out, &target).unwrap();
        let grads = loss.backward().unwrap();
        for p in model.parameters() {
            assert!(
                grads.get(p.as_tensor()).is_some(),
                "missing gradient for LLM parameter"
            );
        }
    }

    // ---------------- Part III: Training & Generation ----------------

    #[test]
    fn submit_cross_entropy_loss() {
        let device = Device::Cpu;
        let logits = Tensor::new(
            &[[2.0_f32, 1.0, 0.0], [0.0, 2.0, 1.0], [1.5, -0.5, 0.25]],
            &device,
        )
        .unwrap();
        let y = Tensor::new(&[0_u32, 2, 1], &device).unwrap();
        let loss = cross_entropy_loss(&logits, &y).unwrap();
        assert_eq!(loss.rank(), 0);

        // Analytical f64 ground truth:
        let l0 = (2.0_f64.exp() + 1.0_f64.exp() + 0.0_f64.exp()).ln() - 2.0;
        let l1 = (0.0_f64.exp() + 2.0_f64.exp() + 1.0_f64.exp()).ln() - 1.0;
        let l2 = (1.5_f64.exp() + (-0.5_f64).exp() + 0.25_f64.exp()).ln() - (-0.5);
        let expected_2d = ((l0 + l1 + l2) / 3.0) as f32;
        assert!(
            (loss.to_scalar::<f32>().unwrap() - expected_2d).abs() < 1e-5,
            "got {}, expected {expected_2d}",
            loss.to_scalar::<f32>().unwrap()
        );

        // Numerical stability check: adding +1000 to all logits must preserve exact loss!
        let shifted = (&logits + 1000.0).unwrap();
        let shifted_loss = cross_entropy_loss(&shifted, &y)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(shifted_loss.is_finite());
        assert!((shifted_loss - expected_2d).abs() < 1e-4);

        let logits3 = Tensor::new(
            &[
                [[1.0_f32, 0.0], [0.5, -0.5]],
                [[-1.0, 2.0], [3.0, 1.0]],
            ],
            &device,
        )
        .unwrap();
        let y3 = Tensor::new(&[[0_u32, 1], [1, 0]], &device).unwrap();
        let a0 = (1.0_f64.exp() + 1.0).ln() - 1.0;
        let a1 = (0.5_f64.exp() + (-0.5_f64).exp()).ln() - (-0.5);
        let a2 = ((-1.0_f64).exp() + 2.0_f64.exp()).ln() - 2.0;
        let a3 = (3.0_f64.exp() + 1.0_f64.exp()).ln() - 3.0;
        let expected_3d = ((a0 + a1 + a2 + a3) / 4.0) as f32;
        let loss3 = cross_entropy_loss(&logits3, &y3)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!((loss3 - expected_3d).abs() < 1e-5);
    }

    #[test]
    fn submit_pretokenize_data() {
        let tokenizer = ChunkTokenizer::new();
        let dir = TempDir::new("pretokenize");
        let in_path = dir.join("sample.txt");
        let out_path = dir.join("sample.bin");

        fs::write(&in_path, "hello!").unwrap();
        pretokenize_data(&tokenizer, &in_path, &out_path, 2, Some(3)).unwrap();

        assert_eq!(
            read_u16_tokens(&out_path),
            vec![104, 101, 108, 108, 111, 33]
        );
        assert_eq!(
            *tokenizer.calls.borrow(),
            vec!["he".to_string(), "ll".to_string(), "o!".to_string()]
        );

        // Edge case: max_chunks = None with multi-byte UTF-8 characters crossing chunk boundaries.
        let tok2 = ChunkTokenizer::new();
        let utf8_in = dir.join("utf8.txt");
        let utf8_out = dir.join("utf8.bin");
        fs::write(&utf8_in, "aé€b").unwrap();
        pretokenize_data(&tok2, &utf8_in, &utf8_out, 2, None).unwrap();
        assert_eq!(
            *tok2.calls.borrow(),
            vec!["aé".to_string(), "€b".to_string()]
        );
        assert_eq!(
            read_u16_tokens(&utf8_out),
            vec!['a' as u16, 'é' as u16, '€' as u16, 'b' as u16]
        );

        // Verify Gpt2Tokenizer encodes special `<|endoftext|>` token properly.
        let gpt2 = Gpt2Tokenizer::new().unwrap();
        assert_eq!(gpt2.eot_token(), 50_256);
        assert_eq!(padded_vocab_size(gpt2.n_vocab()), 50_432);
        let ids = gpt2.encode("Hi!<|endoftext|>");
        assert_eq!(*ids.last().unwrap(), gpt2.eot_token());
    }

    #[test]
    fn submit_dataloader() {
        let dir = TempDir::new("dataloader");
        let bin_path = dir.join("tokens.bin");
        write_u16_tokens(&bin_path, &(0u16..20).collect::<Vec<_>>());

        let loader = DataLoader::new(&bin_path, 3, 2, Device::Cpu).unwrap();
        let batches: Vec<(Tensor, Tensor)> = loader.iter().map(|r| r.unwrap()).collect();
        assert_eq!(batches.len(), 3);
        assert_eq!(
            batches[0].0.to_vec2::<u32>().unwrap(),
            vec![vec![0, 1, 2], vec![4, 5, 6]]
        );
        assert_eq!(
            batches[0].1.to_vec2::<u32>().unwrap(),
            vec![vec![1, 2, 3], vec![5, 6, 7]]
        );
        assert_eq!(
            batches[1].0.to_vec2::<u32>().unwrap(),
            vec![vec![6, 7, 8], vec![10, 11, 12]]
        );
        assert_eq!(
            batches[1].1.to_vec2::<u32>().unwrap(),
            vec![vec![7, 8, 9], vec![11, 12, 13]]
        );
        assert_eq!(
            batches[2].0.to_vec2::<u32>().unwrap(),
            vec![vec![12, 13, 14], vec![16, 17, 18]]
        );
        assert_eq!(
            batches[2].1.to_vec2::<u32>().unwrap(),
            vec![vec![13, 14, 15], vec![17, 18, 19]]
        );

        // Edge case: file too small for even 1 complete batch yields 0 batches.
        let tiny_path = dir.join("tiny.bin");
        write_u16_tokens(&tiny_path, &[1, 2, 3, 4, 5]);
        let empty_loader = DataLoader::new(&tiny_path, 3, 2, Device::Cpu).unwrap();
        assert!(empty_loader.is_empty());
        assert_eq!(empty_loader.iter().count(), 0);
    }

    #[test]
    fn submit_adam() {
        let device = Device::Cpu;
        let w = Var::new(&[[1.0_f32, -2.0], [0.5, 3.0]], &device).unwrap();
        let b = Var::new(&[0.25_f32, -0.75], &device).unwrap();
        let unused = Var::new(&[42.0_f32], &device).unwrap();
        let mut opt =
            Adam::new(vec![w.clone(), b.clone(), unused.clone()], 0.1, (0.8, 0.9), 1e-6).unwrap();

        let step_with_grads = |opt: &mut Adam, gw: &Tensor, gb: &Tensor| {
            // Exact synthetic loss whose gradients wrt w and b are gw and gb.
            let loss = ((w.as_tensor() * gw).unwrap().sum_all().unwrap()
                + (b.as_tensor() * gb).unwrap().sum_all().unwrap())
            .unwrap();
            let grads = loss.backward().unwrap();
            opt.step(&grads).unwrap();
        };

        let gw1 = Tensor::new(&[[0.1_f32, -0.4], [-0.2, 0.3]], &device).unwrap();
        let gb1 = Tensor::new(&[0.5_f32, -0.25], &device).unwrap();
        step_with_grads(&mut opt, &gw1, &gb1);

        let gw2 = Tensor::new(&[[0.2_f32, 0.1], [-0.1, -0.3]], &device).unwrap();
        let gb2 = Tensor::new(&[-0.5_f32, 0.5], &device).unwrap();
        step_with_grads(&mut opt, &gw2, &gb2);

        // Compute exact f64 analytical Adam ground truth.
        fn analytical_adam_2steps(
            init: &[f64],
            g1: &[f64],
            g2: &[f64],
            lr: f64,
            b1: f64,
            b2: f64,
            eps: f64,
        ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
            let mut p = init.to_vec();
            let mut m = vec![0.0_f64; init.len()];
            let mut v = vec![0.0_f64; init.len()];
            for (t, g) in [g1, g2].iter().enumerate() {
                let step_t = (t + 1) as i32;
                let bc1 = 1.0 - b1.powi(step_t);
                let bc2 = 1.0 - b2.powi(step_t);
                for i in 0..p.len() {
                    m[i] = b1 * m[i] + (1.0 - b1) * g[i];
                    v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
                    let mh = m[i] / bc1;
                    let vh = v[i] / bc2;
                    p[i] -= lr * mh / (vh.sqrt() + eps);
                }
            }
            (
                p.into_iter().map(|x| x as f32).collect(),
                m.into_iter().map(|x| x as f32).collect(),
                v.into_iter().map(|x| x as f32).collect(),
            )
        }

        let (exp_w, exp_mw, _) = analytical_adam_2steps(
            &[1.0, -2.0, 0.5, 3.0],
            &[0.1, -0.4, -0.2, 0.3],
            &[0.2, 0.1, -0.1, -0.3],
            0.1,
            0.8,
            0.9,
            1e-6,
        );
        let (exp_b, _, exp_vb) = analytical_adam_2steps(
            &[0.25, -0.75],
            &[0.5, -0.25],
            &[-0.5, 0.5],
            0.1,
            0.8,
            0.9,
            1e-6,
        );

        assert_allclose(
            w.as_tensor(),
            &Tensor::from_vec(exp_w, (2, 2), &device).unwrap(),
            1e-5,
            1e-5,
        );
        assert_allclose(
            b.as_tensor(),
            &Tensor::from_vec(exp_b, (2,), &device).unwrap(),
            1e-5,
            1e-5,
        );
        assert_allclose(
            &opt.m[&w.id()],
            &Tensor::from_vec(exp_mw, (2, 2), &device).unwrap(),
            1e-5,
            1e-5,
        );
        assert_allclose(
            &opt.v[&b.id()],
            &Tensor::from_vec(exp_vb, (2,), &device).unwrap(),
            1e-5,
            1e-5,
        );
        // Parameter absent from GradStore must remain untouched.
        assert_allclose(
            unused.as_tensor(),
            &Tensor::new(&[42.0_f32], &device).unwrap(),
            1e-6,
            1e-6,
        );
    }

    #[test]
    fn submit_train_llm() {
        let device = Device::Cpu;
        let mut model = TinyTrainModel::new_seeded(5, 4, 240);
        let mut ref_model = model.clone_fresh();

        let mut opt = SimpleSgd {
            params: model.parameters(),
            lr: 0.1,
        };
        let mut ref_opt = SimpleSgd {
            params: ref_model.parameters(),
            lr: 0.1,
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

        train_llm(
            &mut model,
            batches.iter().map(|(x, y)| Ok((x.clone(), y.clone()))),
            &mut opt,
        )
        .unwrap();

        for (x, y) in &batches {
            let logits = ref_model.forward(x, 0, false).unwrap();
            let loss = cross_entropy_loss(&logits, y).unwrap();
            let grads = loss.backward().unwrap();
            ref_opt.step(&grads).unwrap();
        }

        assert_allclose(
            &model.embed.weight.as_tensor().narrow(0, 0, 2).unwrap(),
            &ref_model.embed.weight.as_tensor().narrow(0, 0, 2).unwrap(),
            1e-5,
            1e-5,
        );
        assert_allclose(
            &model.proj.weight.as_tensor().narrow(0, 0, 2).unwrap(),
            &ref_model.proj.weight.as_tensor().narrow(0, 0, 2).unwrap(),
            1e-5,
            1e-5,
        );

        let probe = Tensor::new(&[[0_u32, 1, 2]], &device).unwrap();
        assert_allclose(
            &model.forward(&probe, 0, false).unwrap(),
            &ref_model.forward(&probe, 0, false).unwrap(),
            1e-5,
            1e-5,
        );
    }

    #[test]
    fn submit_generate() {
        let mut model = ToyGenerationModel::new(vec![3, 5, 4], 6);
        let tokenizer = ToyGenerationTokenizer::new(4);
        let mut out = Vec::new();

        let generated =
            generate_with_writer(&mut model, &[1, 2], &tokenizer, 0.7, 6, Some(&mut out)).unwrap();

        assert_eq!(generated, vec![3, 5, 4]);
        assert_eq!(String::from_utf8(out).unwrap(), "AB!");
        let seq_positions: Vec<usize> = model.calls.iter().map(|(_, pos, _)| *pos).collect();
        assert_eq!(seq_positions, vec![0, 2, 3]);

        // Edge case: max_tokens truncation when EOT is never sampled.
        let mut model_trunc = ToyGenerationModel::new(vec![3, 5, 3, 5], 6);
        let mut out_trunc = Vec::new();
        let gen_trunc = generate_with_writer(
            &mut model_trunc,
            &[1],
            &tokenizer,
            0.7,
            2,
            Some(&mut out_trunc),
        )
        .unwrap();
        assert_eq!(gen_trunc, vec![3, 5]);
        assert_eq!(String::from_utf8(out_trunc).unwrap(), "AB");
        assert_eq!(model_trunc.calls.len(), 2);

        // Edge case: max_tokens == 0 returns empty without calling model.
        let mut model_zero = ToyGenerationModel::new(vec![3], 6);
        let gen_zero =
            generate_with_writer(&mut model_zero, &[1, 2], &tokenizer, 0.7, 0, None).unwrap();
        assert!(gen_zero.is_empty());
        assert!(model_zero.calls.is_empty());
    }

    #[test]
    fn submit_eval_llm_synthetic() {
        let device = Device::Cpu;
        let dir = TempDir::new("eval_submit");
        let bin_path = dir.join("pattern.bin");

        // Deterministic repeating sequence over 12 tokens.
        let seq: Vec<u16> = (0..640).map(|i| (i % 12) as u16).collect();
        write_u16_tokens(&bin_path, &seq);

        let loader = DataLoader::new(&bin_path, 12, 4, device.clone()).unwrap();
        let mut model = LLM::new(12, 16, 4, 24, 32, 2, &device).unwrap();
        let mut opt = Adam::new(model.parameters(), 2e-2, (0.9, 0.95), 1e-8).unwrap();

        for _ in 0..8 {
            train_llm(&mut model, &loader, &mut opt).unwrap();
        }

        let eval_ids: Vec<u32> = (0..12).collect();
        let eval_tokens = Tensor::from_slice(&eval_ids, (1, 12), &device).unwrap();
        let logits_ok = model
            .forward(&eval_tokens.narrow(1, 0, 11).unwrap(), 0, false)
            .unwrap();
        let phrase_loss = cross_entropy_loss(&logits_ok, &eval_tokens.narrow(1, 1, 11).unwrap())
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        let mut rev_ids = eval_ids.clone();
        rev_ids[1..].reverse();
        let corrupted = Tensor::from_slice(&rev_ids, (1, 12), &device).unwrap();
        let logits_bad = model
            .forward(&corrupted.narrow(1, 0, 11).unwrap(), 0, false)
            .unwrap();
        let corrupted_loss = cross_entropy_loss(&logits_bad, &corrupted.narrow(1, 1, 11).unwrap())
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert!(phrase_loss < 1.0, "expected low phrase_loss, got {phrase_loss}");
        assert!(phrase_loss < corrupted_loss);

        model
            .forward(&eval_tokens.narrow(1, 0, 10).unwrap(), 0, true)
            .unwrap();
        let tail = model
            .forward(&eval_tokens.narrow(1, 10, 1).unwrap(), 10, true)
            .unwrap();
        assert_allclose(&logits_ok.narrow(1, 10, 1).unwrap(), &tail, 3e-4, 3e-4);
    }
}
