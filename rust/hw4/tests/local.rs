//! Local sanity-check tests for Homework 4.
//!
//! Ported directly from `homework/hw4/hw4_tests.py` (`test_*` functions).
//! Run with: `cargo test -p hw4 --test local`
//!
//! Note: `test_eval_llama3_pretrained` downloads the ~4.9 GB Llama 3.2 1B Simplified
//! checkpoint and is marked `#[ignore]` so the default suite runs offline in < 1s.
//! Run it explicitly with: `cargo test --release -p hw4 --test local -- --ignored`

mod local {
    use std::collections::{HashMap, HashSet};

    use candle_core::{D, DType, Device, Result, Tensor};
    use common::{assert_allclose, seeded_randn};
    use hw4::{
        CausalLM, Embedding, GatedMLP, Linear, Llama3Simplified, LlamaConfig, MultiHeadAttention,
        MultiHeadAttentionKVCache, RMSNorm, Tokenizer, TransformerBlock, generate,
        generate_with_writer, load_pretrained_llama3, self_attention, silu,
    };

    fn causal_mask(length: usize, device: &Device) -> Tensor {
        let mut data = vec![0.0_f32; length * length];
        for r in 0..length {
            for c in (r + 1)..length {
                data[r * length + c] = f32::NEG_INFINITY;
            }
        }
        Tensor::from_vec(data, (length, length), device).unwrap()
    }

    fn zero_linear(linear: &Linear) {
        let zeros = Tensor::zeros_like(linear.weight.as_tensor()).unwrap();
        linear.weight.set(&zeros).unwrap();
    }

    fn identity_linear(linear: &Linear) {
        let (rows, cols) = linear.weight.dims2().unwrap();
        let mut data = vec![0.0_f32; rows * cols];
        for i in 0..rows.min(cols) {
            data[i * cols + i] = 1.0;
        }
        let eye = Tensor::from_vec(data, (rows, cols), linear.weight.device()).unwrap();
        linear.weight.set(&eye).unwrap();
    }

    /// Independent reference scaled dot-product attention (in f64 via CPU slices or Candle ops).
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

    /// Independent reference multi-head attention using the given projection weights.
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
        let q = x
            .broadcast_matmul(&wq.t().unwrap())
            .unwrap()
            .reshape((b, s, n_heads, hd))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let k = x
            .broadcast_matmul(&wk.t().unwrap())
            .unwrap()
            .reshape((b, s, n_heads, hd))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let v = x
            .broadcast_matmul(&wv.t().unwrap())
            .unwrap()
            .reshape((b, s, n_heads, hd))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let out = ref_sdpa(&q, &k, &v, mask)
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((b, s, d))
            .unwrap();
        out.broadcast_matmul(&wp.t().unwrap()).unwrap()
    }

    struct ToyTokenizer {
        stop_tokens: HashSet<u32>,
        vocab: HashMap<u32, &'static str>,
    }

    impl ToyTokenizer {
        fn new(stop_tokens: &[u32]) -> Self {
            let mut vocab = HashMap::new();
            vocab.insert(3, "A");
            vocab.insert(4, "!");
            vocab.insert(5, "B");
            Self {
                stop_tokens: stop_tokens.iter().copied().collect(),
                vocab,
            }
        }
    }

    impl Tokenizer for ToyTokenizer {
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

        fn is_stop_token(&self, token: u32) -> bool {
            self.stop_tokens.contains(&token)
        }
    }

    struct ToyModel {
        next_tokens: Vec<u32>,
        vocab_size: usize,
        calls: Vec<(Vec<Vec<u32>>, usize, bool)>,
    }

    impl ToyModel {
        fn new(next_tokens: Vec<u32>, vocab_size: usize) -> Self {
            Self {
                next_tokens,
                vocab_size,
                calls: Vec::new(),
            }
        }
    }

    impl CausalLM for ToyModel {
        fn forward(&mut self, tokens: &Tensor, seq_pos: usize, use_kv_cache: bool) -> Result<Tensor> {
            let token_rows = tokens.to_vec2::<u32>()?;
            let seq_len = tokens.dim(1)?;
            self.calls.push((token_rows, seq_pos, use_kv_cache));

            let mut logits = vec![-1e9_f32; seq_len * self.vocab_size];
            let next_token = self.next_tokens[self.calls.len() - 1] as usize;
            logits[(seq_len - 1) * self.vocab_size + next_token] = 0.0;
            Tensor::from_vec(logits, (1, seq_len, self.vocab_size), tokens.device())
        }
    }

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
        let ref2d = x2d.matmul(&w_t).unwrap();
        assert_allclose(&out2d, &ref2d, 1e-5, 1e-5);

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
    }

    #[test]
    fn test_silu() {
        let device = Device::Cpu;
        let ref_silu = |x: &Tensor| -> Tensor {
            let vals = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let out: Vec<f32> = vals
                .into_iter()
                .map(|v| {
                    let vf = v as f64;
                    (vf / (1.0 + (-vf).exp())) as f32
                })
                .collect();
            Tensor::from_vec(out, x.shape(), &device).unwrap()
        };

        let x2d = seeded_randn(&[10, 20], 20, &device).unwrap();
        assert_allclose(&silu(&x2d).unwrap(), &ref_silu(&x2d), 1e-5, 1e-6);

        let x4d = seeded_randn(&[3, 4, 5, 6], 21, &device).unwrap();
        assert_allclose(&silu(&x4d).unwrap(), &ref_silu(&x4d), 1e-5, 1e-6);
    }

    #[test]
    fn test_rms_norm() {
        let device = Device::Cpu;
        let layer = RMSNorm::new(20, 1e-3, &device).unwrap();
        assert_eq!(layer.weight.dims(), &[20]);
        assert!((layer.eps - 1e-3).abs() < 1e-12);
        let ones = Tensor::ones((20,), DType::F32, &device).unwrap();
        assert_allclose(layer.weight.as_tensor(), &ones, 1e-6, 1e-6);

        let w = (ones + (seeded_randn(&[20], 30, &device).unwrap() * 0.1).unwrap()).unwrap();
        layer.weight.set(&w).unwrap();
        let w_vec = w.to_vec1::<f32>().unwrap();

        let ref_rms = |x: &Tensor| -> Tensor {
            let dims = x.dims();
            let d = *dims.last().unwrap();
            let flat = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let mut out = Vec::with_capacity(flat.len());
            for row in flat.chunks_exact(d) {
                let mean_sq = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / d as f64;
                let denom = (mean_sq + 1e-3).sqrt();
                for (j, &v) in row.iter().enumerate() {
                    out.push(((v as f64 / denom) * w_vec[j] as f64) as f32);
                }
            }
            Tensor::from_vec(out, dims, &device).unwrap()
        };

        let x2d = seeded_randn(&[100, 20], 31, &device).unwrap();
        assert_allclose(&layer.forward(&x2d).unwrap(), &ref_rms(&x2d), 1e-5, 1e-5);

        let x3d = seeded_randn(&[10, 7, 20], 32, &device).unwrap();
        assert_allclose(&layer.forward(&x3d).unwrap(), &ref_rms(&x3d), 1e-5, 1e-5);
    }

    #[test]
    fn test_self_attention() {
        let device = Device::Cpu;
        let mask = causal_mask(5, &device);

        let q = seeded_randn(&[5, 8], 40, &device).unwrap();
        let k = seeded_randn(&[5, 8], 41, &device).unwrap();
        let v = seeded_randn(&[5, 6], 42, &device).unwrap();
        let ref_out = ref_sdpa(&q, &k, &v, Some(&mask));
        let out = self_attention(&q, &k, &v, Some(&mask)).unwrap();
        assert_eq!(out.dims(), &[5, 6]);
        assert_allclose(&out, &ref_out, 1e-5, 1e-5);

        let q4 = seeded_randn(&[2, 3, 5, 8], 43, &device).unwrap();
        let k4 = seeded_randn(&[2, 3, 5, 8], 44, &device).unwrap();
        let v4 = seeded_randn(&[2, 3, 5, 4], 45, &device).unwrap();
        let ref_out4 = ref_sdpa(&q4, &k4, &v4, Some(&mask));
        let out4 = self_attention(&q4, &k4, &v4, Some(&mask)).unwrap();
        assert_eq!(out4.dims(), &[2, 3, 5, 4]);
        assert_allclose(&out4, &ref_out4, 1e-5, 1e-5);
    }

    #[test]
    fn test_multi_head_attention() {
        let device = Device::Cpu;
        let attn = MultiHeadAttention::new(12, 3, &device).unwrap();
        let wq = seeded_randn(&[12, 12], 50, &device).unwrap();
        let wk = seeded_randn(&[12, 12], 51, &device).unwrap();
        let wv = seeded_randn(&[12, 12], 52, &device).unwrap();
        let wp = seeded_randn(&[12, 12], 53, &device).unwrap();
        attn.wq.weight.set(&wq).unwrap();
        attn.wk.weight.set(&wk).unwrap();
        attn.wv.weight.set(&wv).unwrap();
        attn.wp.weight.set(&wp).unwrap();

        let x = seeded_randn(&[2, 5, 12], 54, &device).unwrap();
        let ref_unmasked = ref_mha(&x, &wq, &wk, &wv, &wp, 3, None);
        let out_unmasked = attn.forward(&x, None, 0, false).unwrap();
        assert_eq!(out_unmasked.dims(), &[2, 5, 12]);
        assert_allclose(&out_unmasked, &ref_unmasked, 1e-5, 1e-5);

        let mask = causal_mask(5, &device);
        let ref_masked = ref_mha(&x, &wq, &wk, &wv, &wp, 3, Some(&mask));
        let out_masked = attn.forward(&x, Some(&mask), 0, false).unwrap();
        assert_allclose(&out_masked, &ref_masked, 1e-5, 1e-5);
    }

    #[test]
    fn test_multi_head_attention_kv_cache() {
        let device = Device::Cpu;
        let mut attn = MultiHeadAttentionKVCache::new(12, 3, 8, &device).unwrap();
        assert_eq!(attn.k_cache.dims(), &[1, 8, 12]);
        assert_eq!(attn.v_cache.dims(), &[1, 8, 12]);

        let wq = seeded_randn(&[12, 12], 60, &device).unwrap();
        let wk = seeded_randn(&[12, 12], 61, &device).unwrap();
        let wv = seeded_randn(&[12, 12], 62, &device).unwrap();
        let wp = seeded_randn(&[12, 12], 63, &device).unwrap();
        attn.wq.weight.set(&wq).unwrap();
        attn.wk.weight.set(&wk).unwrap();
        attn.wv.weight.set(&wv).unwrap();
        attn.wp.weight.set(&wp).unwrap();

        let x = seeded_randn(&[1, 5, 12], 64, &device).unwrap();
        let mask = causal_mask(5, &device);

        let ref_out = ref_mha(&x, &wq, &wk, &wv, &wp, 3, Some(&mask));
        let out = attn.forward(&x, Some(&mask), 0, false).unwrap();
        assert_eq!(out.dims(), &[1, 5, 12]);
        assert_allclose(&out, &ref_out, 1e-5, 1e-5);

        let x_prefix = x.narrow(1, 0, 3).unwrap();
        let x_tail = x.narrow(1, 3, 2).unwrap();
        let m_prefix = mask.narrow(0, 0, 3).unwrap().narrow(1, 0, 3).unwrap();
        let m_tail = mask.narrow(0, 3, 2).unwrap();

        let prefix = attn.forward(&x_prefix, Some(&m_prefix), 0, true).unwrap();
        let tail = attn.forward(&x_tail, Some(&m_tail), 3, true).unwrap();
        assert_allclose(&prefix, &ref_out.narrow(1, 0, 3).unwrap(), 1e-5, 1e-5);
        assert_allclose(&tail, &ref_out.narrow(1, 3, 2).unwrap(), 1e-5, 1e-5);

        let k_expected = attn.wk.forward(&x).unwrap();
        let v_expected = attn.wv.forward(&x).unwrap();
        assert_allclose(&attn.k_cache.narrow(1, 0, 5).unwrap(), &k_expected, 1e-5, 1e-5);
        assert_allclose(&attn.v_cache.narrow(1, 0, 5).unwrap(), &v_expected, 1e-5, 1e-5);
    }

    #[test]
    fn test_gated_mlp() {
        let device = Device::Cpu;
        let mlp = GatedMLP::new(2, 3, &device).unwrap();
        assert_eq!(mlp.w1.weight.dims(), &[3, 2]);
        assert_eq!(mlp.w2.weight.dims(), &[2, 3]);
        assert_eq!(mlp.w3.weight.dims(), &[3, 2]);

        mlp.w1
            .weight
            .set(&Tensor::new(&[[1.0_f32, 0.0], [0.0, 1.0], [1.0, -1.0]], &device).unwrap())
            .unwrap();
        mlp.w2
            .weight
            .set(&Tensor::new(&[[1.0_f32, 0.0, -1.0], [0.0, 1.0, 1.0]], &device).unwrap())
            .unwrap();
        mlp.w3
            .weight
            .set(&Tensor::new(&[[1.0_f32, 1.0], [2.0, 0.0], [0.0, -1.0]], &device).unwrap())
            .unwrap();

        let x = Tensor::new(
            &[[0.0_f32, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            &device,
        )
        .unwrap();
        let expected = Tensor::new(
            &[
                [0.0_f32, 0.0],
                [0.7310586, 0.0],
                [-0.26894143, 0.26894143],
                [1.4621172, 1.4621172],
            ],
            &device,
        )
        .unwrap();
        assert_allclose(&mlp.forward(&x).unwrap(), &expected, 1e-5, 1e-5);

        let x3d = seeded_randn(&[2, 4, 2], 70, &device).unwrap();
        assert_eq!(mlp.forward(&x3d).unwrap().dims(), &[2, 4, 2]);
    }

    #[test]
    fn test_transformer_block() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[[1.0_f32, 1.0], [1.0, -1.0]]], &device).unwrap();

        // Case 1: zero attention, active GatedMLP
        let mut block = TransformerBlock::new(2, 1, 3, 4, &device).unwrap();
        block.norm1.eps = 0.0;
        block.norm2.eps = 0.0;
        zero_linear(&block.attn.wq);
        zero_linear(&block.attn.wk);
        zero_linear(&block.attn.wv);
        zero_linear(&block.attn.wp);
        block
            .mlp
            .w1
            .weight
            .set(&Tensor::new(&[[1.0_f32, 0.0], [0.0, 1.0], [1.0, -1.0]], &device).unwrap())
            .unwrap();
        block
            .mlp
            .w2
            .weight
            .set(&Tensor::new(&[[1.0_f32, 0.0, -1.0], [0.0, 1.0, 1.0]], &device).unwrap())
            .unwrap();
        block
            .mlp
            .w3
            .weight
            .set(&Tensor::new(&[[1.0_f32, 1.0], [2.0, 0.0], [0.0, -1.0]], &device).unwrap())
            .unwrap();

        let expected1 = Tensor::new(
            &[[[2.4621172_f32, 2.4621172], [-0.76159406, 0.22371125]]],
            &device,
        )
        .unwrap();
        assert_allclose(&block.forward(&x, None, 0, false).unwrap(), &expected1, 1e-5, 1e-5);

        // Case 2: identity attention with causal mask, zero MLP
        let mut block = TransformerBlock::new(2, 1, 3, 4, &device).unwrap();
        block.norm1.eps = 0.0;
        block.norm2.eps = 0.0;
        identity_linear(&block.attn.wq);
        identity_linear(&block.attn.wk);
        identity_linear(&block.attn.wv);
        identity_linear(&block.attn.wp);
        zero_linear(&block.mlp.w1);
        zero_linear(&block.mlp.w2);
        zero_linear(&block.mlp.w3);

        let mask = causal_mask(2, &device);
        let expected2 =
            Tensor::new(&[[[2.0_f32, 2.0], [2.0, -1.6088594]]], &device).unwrap();
        assert_allclose(
            &block.forward(&x, Some(&mask), 0, false).unwrap(),
            &expected2,
            1e-5,
            1e-5,
        );
    }

    #[test]
    fn test_llama3_simplified() {
        let device = Device::Cpu;
        let mut model = Llama3Simplified::new(5, 4, 2, 6, 6, 1, &device).unwrap();
        assert_eq!(model.embedding.weight.dims(), &[5, 4]);
        assert_eq!(model.pos_embeddings.dims(), &[6, 4]);
        assert_eq!(model.layers.len(), 1);
        assert_eq!(model.output.weight.dims(), &[5, 4]);
        assert_eq!(model.mask.dims(), &[6, 6]);

        let mask_vals = model.mask.to_vec2::<f32>().unwrap();
        assert_eq!(mask_vals[0][0], 0.0);
        assert!(mask_vals[0][1].is_infinite() && mask_vals[0][1].is_sign_negative());
        assert_eq!(mask_vals[1][0], 0.0);

        model
            .embedding
            .weight
            .set(&Tensor::zeros((5, 4), DType::F32, &device).unwrap())
            .unwrap();
        model.pos_embeddings = Tensor::new(
            &[
                [1.0_f32, 1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0, -1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            &device,
        )
        .unwrap();
        model.norm.eps = 0.0;
        model
            .output
            .weight
            .set(
                &Tensor::new(
                    &[
                        [1.0_f32, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                    ],
                    &device,
                )
                .unwrap(),
            )
            .unwrap();
        for layer in &model.layers {
            zero_linear(&layer.attn.wq);
            zero_linear(&layer.attn.wk);
            zero_linear(&layer.attn.wv);
            zero_linear(&layer.attn.wp);
            zero_linear(&layer.mlp.w1);
            zero_linear(&layer.mlp.w2);
            zero_linear(&layer.mlp.w3);
        }

        let tokens = Tensor::new(&[[2_u32, 3, 4]], &device).unwrap();
        let out = model.forward(&tokens, 0, false).unwrap();
        let expected = Tensor::new(
            &[[
                [1.0_f32, 1.0, 1.0, 1.0, 4.0],
                [1.0, -1.0, 1.0, -1.0, 0.0],
                [-1.0, 1.0, -1.0, 1.0, 0.0],
            ]],
            &device,
        )
        .unwrap();
        assert_eq!(out.dims(), &[1, 3, 5]);
        assert_allclose(&out, &expected, 1e-5, 1e-5);
    }

    #[test]
    fn test_eval_llama3_synthetic() {
        let device = Device::Cpu;
        // Verify params.json deserialization against the exact Llama 3.2 1B Simplified schema.
        let cfg: LlamaConfig = serde_json::from_str(
            r#"{
                "dim": 2048,
                "n_layers": 16,
                "n_heads": 32,
                "vocab_size": 128256,
                "ffn_dim_multiplier": 4,
                "norm_eps": 1e-05,
                "max_seq_len": 4096
            }"#,
        )
        .unwrap();
        assert_eq!(cfg.ffn_dim(), 8192);

        // Build a small 2-layer Llama checkpoint and verify load_llama_weights + full vs KV-cache equivalence.
        let (vocab, dim, n_heads, max_seq, ffn, n_layers) = (16, 8, 2, 10, 12, 2);
        let mut ckpt = HashMap::new();
        let mut seed = 100u64;
        let mut next = |shape: &[usize]| -> Tensor {
            seed += 1;
            (seeded_randn(shape, seed, &device).unwrap() * 0.25).unwrap()
        };
        ckpt.insert("tok_embeddings.weight".to_string(), next(&[vocab, dim]));
        ckpt.insert("pos_embeddings.weight".to_string(), next(&[max_seq, dim]));
        ckpt.insert("norm.weight".to_string(), Tensor::ones((dim,), DType::F32, &device).unwrap());
        ckpt.insert("output.weight".to_string(), next(&[vocab, dim]));
        for i in 0..n_layers {
            ckpt.insert(format!("layers.{i}.attention.wq.weight"), next(&[dim, dim]));
            ckpt.insert(format!("layers.{i}.attention.wk.weight"), next(&[dim, dim]));
            ckpt.insert(format!("layers.{i}.attention.wv.weight"), next(&[dim, dim]));
            ckpt.insert(format!("layers.{i}.attention.wo.weight"), next(&[dim, dim]));
            ckpt.insert(format!("layers.{i}.feed_forward.w1.weight"), next(&[ffn, dim]));
            ckpt.insert(format!("layers.{i}.feed_forward.w2.weight"), next(&[dim, ffn]));
            ckpt.insert(format!("layers.{i}.feed_forward.w3.weight"), next(&[ffn, dim]));
            ckpt.insert(
                format!("layers.{i}.attention_norm.weight"),
                Tensor::ones((dim,), DType::F32, &device).unwrap(),
            );
            ckpt.insert(
                format!("layers.{i}.ffn_norm.weight"),
                Tensor::ones((dim,), DType::F32, &device).unwrap(),
            );
        }

        let mut model =
            Llama3Simplified::new(vocab, dim, n_heads, max_seq, ffn, n_layers, &device).unwrap();
        model.load_llama_weights(&ckpt).unwrap();

        let tokens = Tensor::new(&[[0_u32, 1, 2, 3]], &device).unwrap();
        let full = model.forward(&tokens, 0, false).unwrap();
        let prefix = model.forward(&tokens.narrow(1, 0, 3).unwrap(), 0, true).unwrap();
        let tail = model.forward(&tokens.narrow(1, 3, 1).unwrap(), 3, true).unwrap();

        assert_eq!(full.dims(), &[1, 4, vocab]);
        assert_eq!(prefix.dims(), &[1, 3, vocab]);
        assert_eq!(tail.dims(), &[1, 1, vocab]);
        assert!(
            full.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
        assert_allclose(&full.narrow(1, 0, 3).unwrap(), &prefix, 1e-4, 1e-4);
        assert_allclose(&full.narrow(1, 3, 1).unwrap(), &tail, 1e-4, 1e-4);
    }

    #[test]
    #[ignore = "downloads ~4.9 GB Llama-3.2-1B-Instruct-Simplified checkpoint from HuggingFace"]
    fn test_eval_llama3_pretrained() {
        let device = Device::Cpu;
        let (mut model, _files) = load_pretrained_llama3(&device).unwrap();
        assert!(!model.layers.is_empty());

        let tokens = Tensor::new(&[[0_u32, 1, 2, 3]], &device).unwrap();
        let full = model.forward(&tokens, 0, false).unwrap();
        let prefix = model.forward(&tokens.narrow(1, 0, 3).unwrap(), 0, true).unwrap();
        let tail = model.forward(&tokens.narrow(1, 3, 1).unwrap(), 3, true).unwrap();

        assert_eq!(full.dim(0).unwrap(), 1);
        assert_eq!(full.dim(1).unwrap(), 4);
        assert_eq!(prefix.dim(1).unwrap(), 3);
        assert_eq!(tail.dim(1).unwrap(), 1);
        assert_eq!(full.dim(2).unwrap(), model.output.weight.dim(0).unwrap());
        assert_allclose(&full.narrow(1, 3, 1).unwrap(), &tail, 3e-4, 3e-4);
    }

    #[test]
    fn test_generate() {
        let mut model = ToyModel::new(vec![3, 4], 6);
        let tokenizer = ToyTokenizer::new(&[4]);
        let mut out = Vec::new();

        let generated =
            generate_with_writer(&mut model, &[1, 2], &tokenizer, 0.7, 5, Some(&mut out)).unwrap();

        assert_eq!(generated, vec![3, 4]);
        assert_eq!(String::from_utf8(out).unwrap(), "A!");
        assert_eq!(model.calls.len(), 2);
        assert_eq!(model.calls[0], (vec![vec![1, 2]], 0, true));
        assert_eq!(model.calls[1], (vec![vec![3]], 2, true));

        // Also check the silent convenience wrapper `generate(..., verbose = false)`.
        let mut model2 = ToyModel::new(vec![3, 4], 6);
        let generated2 = generate(&mut model2, &[1, 2], &tokenizer, 0.7, 5, false).unwrap();
        assert_eq!(generated2, vec![3, 4]);
    }
}
