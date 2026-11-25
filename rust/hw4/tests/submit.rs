//! Grader-equivalent ("submit") tests for Homework 4.
//!
//! Ported from `submit_*` in `homework/hw4/hw4_tests.py` with independent `f64`
//! ground-truth verification plus edge-case checks (causal future-token non-leakage,
//! token-by-token KV-cache equivalence, rectangular `seq_q != seq_k` attention,
//! multi-layer `load_llama_weights`, and generation boundary conditions).
//!
//! Run with: `cargo test -p hw4 --test submit`

mod submit {
    use std::collections::{HashMap, HashSet};

    use candle_core::{DType, Device, Result, Tensor};
    use common::{assert_allclose, seeded_randn};
    use hw4::{
        CausalLM, Embedding, GatedMLP, Linear, Llama3Simplified, MultiHeadAttention,
        MultiHeadAttentionKVCache, RMSNorm, Tokenizer, TransformerBlock, generate_with_writer,
        self_attention, silu,
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

    // ---- Independent f64 reference implementations ----

    fn ref_matmul_2d(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let m = a.len();
        let k = a[0].len();
        let n = b[0].len();
        let mut out = vec![vec![0.0_f64; n]; m];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a[i][p] * b[p][j];
                }
                out[i][j] = sum;
            }
        }
        out
    }

    fn transpose_2d(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let m = a.len();
        let n = a[0].len();
        let mut out = vec![vec![0.0_f64; m]; n];
        for i in 0..m {
            for j in 0..n {
                out[j][i] = a[i][j];
            }
        }
        out
    }

    fn ref_linear_2d(x: &[Vec<f64>], w: &[Vec<f64>]) -> Vec<Vec<f64>> {
        ref_matmul_2d(x, &transpose_2d(w))
    }

    fn ref_silu_scalar(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }

    fn ref_rms_norm_2d(x: &[Vec<f64>], w: &[f64], eps: f64) -> Vec<Vec<f64>> {
        let d = w.len();
        x.iter()
            .map(|row| {
                let mean_sq = row.iter().map(|v| v * v).sum::<f64>() / d as f64;
                let denom = (mean_sq + eps).sqrt();
                row.iter()
                    .enumerate()
                    .map(|(j, &v)| (v / denom) * w[j])
                    .collect()
            })
            .collect()
    }

    fn ref_sdpa_2d(
        q: &[Vec<f64>],
        k: &[Vec<f64>],
        v: &[Vec<f64>],
        mask: Option<&[Vec<f64>]>,
    ) -> Vec<Vec<f64>> {
        let sq = q.len();
        let sk = k.len();
        let d = q[0].len();
        let dv = v[0].len();
        let scale = (d as f64).sqrt();

        let mut probs = vec![vec![0.0_f64; sk]; sq];
        for i in 0..sq {
            let mut scores = vec![0.0_f64; sk];
            for j in 0..sk {
                let mut dot = 0.0;
                for p in 0..d {
                    dot += q[i][p] * k[j][p];
                }
                let mut s = dot / scale;
                if let Some(m) = mask {
                    s += m[i][j];
                }
                scores[j] = s;
            }
            let max_s = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for j in 0..sk {
                probs[i][j] = exps[j] / sum;
            }
        }

        let mut out = vec![vec![0.0_f64; dv]; sq];
        for i in 0..sq {
            for j in 0..dv {
                let mut acc = 0.0;
                for t in 0..sk {
                    acc += probs[i][t] * v[t][j];
                }
                out[i][j] = acc;
            }
        }
        out
    }

    fn ref_mha_2d(
        x: &[Vec<f64>],
        wq: &[Vec<f64>],
        wk: &[Vec<f64>],
        wv: &[Vec<f64>],
        wp: &[Vec<f64>],
        n_heads: usize,
        mask: Option<&[Vec<f64>]>,
    ) -> Vec<Vec<f64>> {
        let s = x.len();
        let d = x[0].len();
        let hd = d / n_heads;
        let q = ref_linear_2d(x, wq);
        let k = ref_linear_2d(x, wk);
        let v = ref_linear_2d(x, wv);

        let mut concat = vec![vec![0.0_f64; d]; s];
        for h in 0..n_heads {
            let qh: Vec<Vec<f64>> = q.iter().map(|r| r[h * hd..(h + 1) * hd].to_vec()).collect();
            let kh: Vec<Vec<f64>> = k.iter().map(|r| r[h * hd..(h + 1) * hd].to_vec()).collect();
            let vh: Vec<Vec<f64>> = v.iter().map(|r| r[h * hd..(h + 1) * hd].to_vec()).collect();
            let oh = ref_sdpa_2d(&qh, &kh, &vh, mask);
            for t in 0..s {
                concat[t][h * hd..(h + 1) * hd].copy_from_slice(&oh[t]);
            }
        }
        ref_linear_2d(&concat, wp)
    }

    fn to_tensor_2d(v: &[Vec<f64>], device: &Device) -> Tensor {
        let r = v.len();
        let c = v[0].len();
        let flat: Vec<f32> = v.iter().flatten().map(|&x| x as f32).collect();
        Tensor::from_vec(flat, (r, c), device).unwrap()
    }

    fn to_tensor_3d(v: &[Vec<Vec<f64>>], device: &Device) -> Tensor {
        let b = v.len();
        let r = v[0].len();
        let c = v[0][0].len();
        let flat: Vec<f32> = v.iter().flatten().flatten().map(|&x| x as f32).collect();
        Tensor::from_vec(flat, (b, r, c), device).unwrap()
    }

    fn causal_mask_f64(n: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|r| {
                (0..n)
                    .map(|c| if c > r { f64::NEG_INFINITY } else { 0.0 })
                    .collect()
            })
            .collect()
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
    fn submit_linear() {
        let device = Device::Cpu;
        let layer = Linear::new(4, 3, &device).unwrap();
        let w = vec![
            vec![1.0, -1.0, 0.5, 2.0],
            vec![-0.5, 0.0, 1.0, -1.5],
            vec![2.0, 1.5, -0.5, 0.25],
        ];
        layer.weight.set(&to_tensor_2d(&w, &device)).unwrap();

        let x = vec![vec![1.0, 2.0, -1.0, 0.5], vec![0.0, -1.0, 2.0, 3.0]];
        let x_t = to_tensor_2d(&x, &device);
        let expected_2d = ref_linear_2d(&x, &w);
        assert_allclose(
            &layer.forward(&x_t).unwrap(),
            &to_tensor_2d(&expected_2d, &device),
            1e-5,
            1e-5,
        );

        let neg_x: Vec<Vec<f64>> = x.iter().map(|r| r.iter().map(|v| -v).collect()).collect();
        let stacked = Tensor::stack(&[&x_t, &x_t.neg().unwrap()], 0).unwrap();
        let expected_3d = vec![expected_2d, ref_linear_2d(&neg_x, &w)];
        assert_allclose(
            &layer.forward(&stacked).unwrap(),
            &to_tensor_3d(&expected_3d, &device),
            1e-5,
            1e-5,
        );
    }

    #[test]
    fn submit_embedding() {
        let device = Device::Cpu;
        let layer = Embedding::new(8, 3, &device).unwrap();
        let w = vec![
            vec![0.0, 0.5, 1.0],
            vec![1.0, -1.0, 0.0],
            vec![0.25, 0.5, 0.75],
            vec![-0.5, 1.5, -1.0],
            vec![2.0, 0.0, 1.0],
            vec![1.25, -0.25, 0.5],
            vec![0.75, 0.25, -0.75],
            vec![-1.0, -0.5, 0.5],
        ];
        layer.weight.set(&to_tensor_2d(&w, &device)).unwrap();

        let y2d = Tensor::new(&[[0_u32, 3, 5], [6, 1, 4]], &device).unwrap();
        let expected_2d = vec![
            vec![w[0].clone(), w[3].clone(), w[5].clone()],
            vec![w[6].clone(), w[1].clone(), w[4].clone()],
        ];
        assert_allclose(
            &layer.forward(&y2d).unwrap(),
            &to_tensor_3d(&expected_2d, &device),
            1e-6,
            1e-6,
        );

        let y1d = Tensor::new(&[7_u32, 2, 0], &device).unwrap();
        let expected_1d = vec![w[7].clone(), w[2].clone(), w[0].clone()];
        assert_allclose(
            &layer.forward(&y1d).unwrap(),
            &to_tensor_2d(&expected_1d, &device),
            1e-6,
            1e-6,
        );

        // Edge case: 3D index tensor [2, 1, 2] -> [2, 1, 2, 3]
        let y3d = Tensor::new(&[[[4_u32, 7]], [[1, 6]]], &device).unwrap();
        let out3d = layer.forward(&y3d).unwrap();
        assert_eq!(out3d.dims(), &[2, 1, 2, 3]);
        let flat_expected: Vec<f32> = [4, 7, 1, 6]
            .iter()
            .flat_map(|&idx| w[idx].iter().map(|&v| v as f32))
            .collect();
        assert_allclose(
            &out3d,
            &Tensor::from_vec(flat_expected, (2, 1, 2, 3), &device).unwrap(),
            1e-6,
            1e-6,
        );
    }

    #[test]
    fn submit_silu() {
        let device = Device::Cpu;
        let x = vec![
            vec![-2.0, -0.5, 0.0, 1.0],
            vec![2.0, 3.0, -1.0, 0.25],
        ];
        let expected: Vec<Vec<f64>> = x
            .iter()
            .map(|r| r.iter().map(|&v| ref_silu_scalar(v)).collect())
            .collect();
        assert_allclose(
            &silu(&to_tensor_2d(&x, &device)).unwrap(),
            &to_tensor_2d(&expected, &device),
            1e-5,
            1e-6,
        );

        let x1 = Tensor::new(&[-3.0_f32, 0.0, 3.0], &device).unwrap();
        let e1 = Tensor::new(
            &[
                ref_silu_scalar(-3.0) as f32,
                0.0_f32,
                ref_silu_scalar(3.0) as f32,
            ],
            &device,
        )
        .unwrap();
        assert_allclose(&silu(&x1).unwrap(), &e1, 1e-5, 1e-6);
    }

    #[test]
    fn submit_rms_norm() {
        let device = Device::Cpu;
        let layer = RMSNorm::new(4, 1e-4, &device).unwrap();
        let w = vec![1.0, 1.5, -0.5, 0.25];
        layer
            .weight
            .set(&Tensor::new(&[1.0_f32, 1.5, -0.5, 0.25], &device).unwrap())
            .unwrap();

        let x = vec![
            vec![1.0, -1.0, 0.5, 0.5],
            vec![2.0, 0.0, -2.0, 1.0],
        ];
        let x_t = to_tensor_2d(&x, &device);
        let expected = ref_rms_norm_2d(&x, &w, 1e-4);
        let expected_t = to_tensor_2d(&expected, &device);

        assert_allclose(&layer.forward(&x_t).unwrap(), &expected_t, 1e-5, 1e-5);
        assert_allclose(
            &layer.forward(&x_t.unsqueeze(0).unwrap()).unwrap(),
            &expected_t.unsqueeze(0).unwrap(),
            1e-5,
            1e-5,
        );

        // Edge case: positive scale invariance when eps = 0.
        let mut norm_inv = RMSNorm::new(4, 0.0, &device).unwrap();
        norm_inv.weight.set(&Tensor::ones((4,), DType::F32, &device).unwrap()).unwrap();
        norm_inv.eps = 0.0;
        let scaled = (&x_t * 7.5).unwrap();
        assert_allclose(
            &norm_inv.forward(&x_t).unwrap(),
            &norm_inv.forward(&scaled).unwrap(),
            1e-5,
            1e-5,
        );
    }

    #[test]
    fn submit_self_attention() {
        let device = Device::Cpu;
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let k = vec![vec![1.0, 0.5], vec![0.0, 1.0], vec![1.0, -1.0]];
        let v = vec![vec![0.5, 1.0], vec![1.5, -0.5], vec![-1.0, 0.25]];
        let mask_f64 = causal_mask_f64(3);

        let qt = to_tensor_2d(&q, &device);
        let kt = to_tensor_2d(&k, &device);
        let vt = to_tensor_2d(&v, &device);
        let mask = causal_mask(3, &device);

        let exp_masked = ref_sdpa_2d(&q, &k, &v, Some(&mask_f64));
        let exp_unmasked = ref_sdpa_2d(&q, &k, &v, None);
        assert_allclose(
            &self_attention(&qt, &kt, &vt, Some(&mask)).unwrap(),
            &to_tensor_2d(&exp_masked, &device),
            1e-5,
            1e-5,
        );
        assert_allclose(
            &self_attention(&qt, &kt, &vt, None).unwrap(),
            &to_tensor_2d(&exp_unmasked, &device),
            1e-5,
            1e-5,
        );

        // Edge case 1: causal mask prevents future-token leakage.
        let mut k_perturbed = k.clone();
        k_perturbed[2] = vec![99.0, -42.0];
        let mut v_perturbed = v.clone();
        v_perturbed[2] = vec![-123.0, 456.0];
        let out_orig = self_attention(&qt, &kt, &vt, Some(&mask)).unwrap();
        let out_pert = self_attention(
            &qt,
            &to_tensor_2d(&k_perturbed, &device),
            &to_tensor_2d(&v_perturbed, &device),
            Some(&mask),
        )
        .unwrap();
        assert_allclose(
            &out_orig.narrow(0, 0, 2).unwrap(),
            &out_pert.narrow(0, 0, 2).unwrap(),
            1e-6,
            1e-6,
        );

        // Edge case 2: rectangular attention (seq_q = 1, seq_k = 3, as during KV-cached decoding).
        let q1 = vec![q[2].clone()];
        let exp_rect = ref_sdpa_2d(&q1, &k, &v, None);
        let out_rect = self_attention(&to_tensor_2d(&q1, &device), &kt, &vt, None).unwrap();
        assert_eq!(out_rect.dims(), &[1, 2]);
        assert_allclose(&out_rect, &to_tensor_2d(&exp_rect, &device), 1e-5, 1e-5);
    }

    #[test]
    fn submit_multi_head_attention() {
        let device = Device::Cpu;
        let attn = MultiHeadAttention::new(4, 2, &device).unwrap();
        let wq = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let wk = vec![
            vec![1.0, 0.0, 0.5, 0.0],
            vec![0.0, 1.0, 0.0, 0.5],
            vec![0.5, 0.0, 1.0, 0.0],
            vec![0.0, 0.5, 0.0, 1.0],
        ];
        let wv = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, -1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, -1.0],
        ];
        let wp = wq.clone();

        attn.wq.weight.set(&to_tensor_2d(&wq, &device)).unwrap();
        attn.wk.weight.set(&to_tensor_2d(&wk, &device)).unwrap();
        attn.wv.weight.set(&to_tensor_2d(&wv, &device)).unwrap();
        attn.wp.weight.set(&to_tensor_2d(&wp, &device)).unwrap();

        let x = vec![
            vec![1.0, 0.0, -1.0, 0.5],
            vec![0.0, 1.0, 0.5, -0.5],
            vec![1.0, 1.0, 0.0, 1.0],
        ];
        let xt = to_tensor_3d(std::slice::from_ref(&x), &device);
        let mask = causal_mask(3, &device);
        let mask_f64 = causal_mask_f64(3);

        let exp_unmasked = ref_mha_2d(&x, &wq, &wk, &wv, &wp, 2, None);
        let exp_masked = ref_mha_2d(&x, &wq, &wk, &wv, &wp, 2, Some(&mask_f64));

        assert_allclose(
            &attn.forward(&xt, None, 0, false).unwrap(),
            &to_tensor_3d(&[exp_unmasked], &device),
            1e-5,
            1e-5,
        );
        assert_allclose(
            &attn.forward(&xt, Some(&mask), 0, false).unwrap(),
            &to_tensor_3d(&[exp_masked], &device),
            1e-5,
            1e-5,
        );
    }

    #[test]
    fn submit_multi_head_attention_kv_cache() {
        let device = Device::Cpu;
        let mut attn = MultiHeadAttentionKVCache::new(4, 2, 6, &device).unwrap();
        let wq = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let wk = vec![
            vec![1.0, 0.5, 0.0, 0.0],
            vec![0.0, 1.0, 0.5, 0.0],
            vec![0.0, 0.0, 1.0, 0.5],
            vec![0.5, 0.0, 0.0, 1.0],
        ];
        let wv = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, -1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, -1.0],
        ];
        let wp = wq.clone();

        attn.wq.weight.set(&to_tensor_2d(&wq, &device)).unwrap();
        attn.wk.weight.set(&to_tensor_2d(&wk, &device)).unwrap();
        attn.wv.weight.set(&to_tensor_2d(&wv, &device)).unwrap();
        attn.wp.weight.set(&to_tensor_2d(&wp, &device)).unwrap();

        let x = vec![
            vec![1.0, 0.0, -1.0, 0.5],
            vec![0.0, 1.0, 0.5, -0.5],
            vec![1.0, 1.0, 0.0, 1.0],
            vec![-1.0, 0.5, 1.0, 0.0],
        ];
        let xt = to_tensor_3d(std::slice::from_ref(&x), &device);
        let mask = causal_mask(4, &device);
        let mask_f64 = causal_mask_f64(4);
        let exp_full = to_tensor_3d(&[ref_mha_2d(&x, &wq, &wk, &wv, &wp, 2, Some(&mask_f64))], &device);

        let full = attn.forward(&xt, Some(&mask), 0, false).unwrap();
        let prefix = attn
            .forward(
                &xt.narrow(1, 0, 2).unwrap(),
                Some(&mask.narrow(0, 0, 2).unwrap().narrow(1, 0, 2).unwrap()),
                0,
                true,
            )
            .unwrap();
        let tail = attn
            .forward(
                &xt.narrow(1, 2, 2).unwrap(),
                Some(&mask.narrow(0, 2, 2).unwrap()),
                2,
                true,
            )
            .unwrap();

        assert_allclose(&full, &exp_full, 1e-5, 1e-5);
        assert_allclose(&prefix, &exp_full.narrow(1, 0, 2).unwrap(), 1e-5, 1e-5);
        assert_allclose(&tail, &exp_full.narrow(1, 2, 2).unwrap(), 1e-5, 1e-5);

        let exp_k = to_tensor_3d(&[ref_linear_2d(&x, &wk)], &device);
        assert_allclose(&attn.k_cache.narrow(1, 0, 4).unwrap(), &exp_k, 1e-5, 1e-5);

        // Edge case: 1-token-at-a-time autoregressive decoding matches full forward pass exactly.
        let mut step_outputs = Vec::new();
        for pos in 0..4 {
            let x_tok = xt.narrow(1, pos, 1).unwrap();
            let m_tok = mask
                .narrow(0, pos, 1)
                .unwrap()
                .narrow(1, 0, pos + 1)
                .unwrap();
            step_outputs.push(attn.forward(&x_tok, Some(&m_tok), pos, true).unwrap());
        }
        let step_refs: Vec<&Tensor> = step_outputs.iter().collect();
        let stitched = Tensor::cat(&step_refs, 1).unwrap();
        assert_allclose(&stitched, &exp_full, 1e-5, 1e-5);
    }

    #[test]
    fn submit_gated_mlp() {
        let device = Device::Cpu;
        let mlp = GatedMLP::new(2, 3, &device).unwrap();
        let w1 = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, -1.0]];
        let w2 = vec![vec![1.0, 0.0, -1.0], vec![0.0, 1.0, 1.0]];
        let w3 = vec![vec![1.0, 1.0], vec![2.0, 0.0], vec![0.0, -1.0]];
        mlp.w1.weight.set(&to_tensor_2d(&w1, &device)).unwrap();
        mlp.w2.weight.set(&to_tensor_2d(&w2, &device)).unwrap();
        mlp.w3.weight.set(&to_tensor_2d(&w3, &device)).unwrap();

        let x = vec![vec![1.0, -1.0], vec![2.0, 0.0], vec![0.5, 0.5]];
        let h1 = ref_linear_2d(&x, &w1);
        let h3 = ref_linear_2d(&x, &w3);
        let gated: Vec<Vec<f64>> = h1
            .iter()
            .zip(h3.iter())
            .map(|(r1, r3)| {
                r1.iter()
                    .zip(r3.iter())
                    .map(|(&a, &b)| ref_silu_scalar(a) * b)
                    .collect()
            })
            .collect();
        let expected = ref_linear_2d(&gated, &w2);

        assert_allclose(
            &mlp.forward(&to_tensor_2d(&x, &device)).unwrap(),
            &to_tensor_2d(&expected, &device),
            1e-5,
            1e-5,
        );
    }

    #[test]
    fn submit_transformer_block() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[[1.0_f32, 1.0], [1.0, -1.0]]], &device).unwrap();

        // Case 1: MLP path only
        let mut block1 = TransformerBlock::new(2, 1, 3, 4, &device).unwrap();
        block1.norm1.eps = 0.0;
        block1.norm2.eps = 0.0;
        zero_linear(&block1.attn.wq);
        zero_linear(&block1.attn.wk);
        zero_linear(&block1.attn.wv);
        zero_linear(&block1.attn.wp);
        block1
            .mlp
            .w1
            .weight
            .set(&Tensor::new(&[[1.0_f32, 0.0], [0.0, 1.0], [1.0, -1.0]], &device).unwrap())
            .unwrap();
        block1
            .mlp
            .w2
            .weight
            .set(&Tensor::new(&[[1.0_f32, 0.0, -1.0], [0.0, 1.0, 1.0]], &device).unwrap())
            .unwrap();
        block1
            .mlp
            .w3
            .weight
            .set(&Tensor::new(&[[1.0_f32, 1.0], [2.0, 0.0], [0.0, -1.0]], &device).unwrap())
            .unwrap();
        let exp1 = Tensor::new(
            &[[[2.4621172_f32, 2.4621172], [-0.76159406, 0.22371125]]],
            &device,
        )
        .unwrap();
        assert_allclose(&block1.forward(&x, None, 0, false).unwrap(), &exp1, 1e-5, 1e-5);

        // Case 2: Attention path only
        let mut block2 = TransformerBlock::new(2, 1, 3, 4, &device).unwrap();
        block2.norm1.eps = 0.0;
        block2.norm2.eps = 0.0;
        identity_linear(&block2.attn.wq);
        identity_linear(&block2.attn.wk);
        identity_linear(&block2.attn.wv);
        identity_linear(&block2.attn.wp);
        zero_linear(&block2.mlp.w1);
        zero_linear(&block2.mlp.w2);
        zero_linear(&block2.mlp.w3);
        let mask2 = causal_mask(2, &device);
        let exp2 = Tensor::new(&[[[2.0_f32, 2.0], [2.0, -1.6088594]]], &device).unwrap();
        assert_allclose(
            &block2.forward(&x, Some(&mask2), 0, false).unwrap(),
            &exp2,
            1e-5,
            1e-5,
        );

        // Edge case: combined non-zero Attention + MLP matches incremental KV-cache decoding.
        let mut block3 = TransformerBlock::new(8, 2, 12, 8, &device).unwrap();
        block3.attn.wq.weight.set(&seeded_randn(&[8, 8], 80, &device).unwrap()).unwrap();
        block3.attn.wk.weight.set(&seeded_randn(&[8, 8], 81, &device).unwrap()).unwrap();
        block3.attn.wv.weight.set(&seeded_randn(&[8, 8], 82, &device).unwrap()).unwrap();
        block3.attn.wp.weight.set(&seeded_randn(&[8, 8], 83, &device).unwrap()).unwrap();
        block3.mlp.w1.weight.set(&seeded_randn(&[12, 8], 84, &device).unwrap()).unwrap();
        block3.mlp.w2.weight.set(&seeded_randn(&[8, 12], 85, &device).unwrap()).unwrap();
        block3.mlp.w3.weight.set(&seeded_randn(&[12, 8], 86, &device).unwrap()).unwrap();

        let x5 = seeded_randn(&[1, 5, 8], 87, &device).unwrap();
        let m5 = causal_mask(5, &device);
        let full = block3.forward(&x5, Some(&m5), 0, false).unwrap();
        let pre = block3
            .forward(
                &x5.narrow(1, 0, 3).unwrap(),
                Some(&m5.narrow(0, 0, 3).unwrap().narrow(1, 0, 3).unwrap()),
                0,
                true,
            )
            .unwrap();
        let tail = block3
            .forward(&x5.narrow(1, 3, 2).unwrap(), Some(&m5.narrow(0, 3, 2).unwrap()), 3, true)
            .unwrap();
        assert_allclose(&pre, &full.narrow(1, 0, 3).unwrap(), 1e-4, 1e-4);
        assert_allclose(&tail, &full.narrow(1, 3, 2).unwrap(), 1e-4, 1e-4);
    }

    #[test]
    fn submit_llama3_simplified() {
        let device = Device::Cpu;
        let mut model = Llama3Simplified::new(5, 4, 2, 6, 6, 1, &device).unwrap();
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
        let expected = Tensor::new(
            &[[
                [1.0_f32, 1.0, 1.0, 1.0, 4.0],
                [1.0, -1.0, 1.0, -1.0, 0.0],
                [-1.0, 1.0, -1.0, 1.0, 0.0],
            ]],
            &device,
        )
        .unwrap();
        assert_allclose(&model.forward(&tokens, 0, false).unwrap(), &expected, 1e-5, 1e-5);

        // Verify exact upper-triangular -inf mask structure on [:4, :4].
        let mask_4x4 = model
            .mask
            .narrow(0, 0, 4)
            .unwrap()
            .narrow(1, 0, 4)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for (r, row) in mask_4x4.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                if c > r {
                    assert!(v.is_infinite() && v.is_sign_negative());
                } else {
                    assert_eq!(v, 0.0);
                }
            }
        }

        // Multi-layer random-weights check: causal non-leakage + token-by-token KV-cache equivalence.
        let (vocab, dim, n_heads, max_seq, ffn, n_layers) = (12, 8, 2, 8, 12, 2);
        let mut ckpt = HashMap::new();
        let mut seed = 200u64;
        let mut next = |shape: &[usize]| -> Tensor {
            seed += 1;
            (seeded_randn(shape, seed, &device).unwrap() * 0.3).unwrap()
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

        let mut lm =
            Llama3Simplified::new(vocab, dim, n_heads, max_seq, ffn, n_layers, &device).unwrap();
        lm.load_llama_weights(&ckpt).unwrap();

        let seq_a = Tensor::new(&[[1_u32, 4, 2, 7, 3]], &device).unwrap();
        let seq_b = Tensor::new(&[[1_u32, 4, 2, 9, 11]], &device).unwrap();
        let out_a = lm.forward(&seq_a, 0, false).unwrap();
        let out_b = lm.forward(&seq_b, 0, false).unwrap();
        // Positions 0, 1, 2 must be identical because positions 3 and 4 are causally masked.
        assert_allclose(
            &out_a.narrow(1, 0, 3).unwrap(),
            &out_b.narrow(1, 0, 3).unwrap(),
            1e-5,
            1e-5,
        );

        // Prefill 2 tokens, then decode 3 tokens one-by-one with KV cache.
        let p0 = lm.forward(&seq_a.narrow(1, 0, 2).unwrap(), 0, true).unwrap();
        let p1 = lm.forward(&seq_a.narrow(1, 2, 1).unwrap(), 2, true).unwrap();
        let p2 = lm.forward(&seq_a.narrow(1, 3, 1).unwrap(), 3, true).unwrap();
        let p3 = lm.forward(&seq_a.narrow(1, 4, 1).unwrap(), 4, true).unwrap();
        let cached_full = Tensor::cat(&[&p0, &p1, &p2, &p3], 1).unwrap();
        assert_allclose(&cached_full, &out_a, 1e-4, 1e-4);
    }

    #[test]
    fn submit_generate() {
        let mut model = ToyModel::new(vec![3, 5, 4], 6);
        let tokenizer = ToyTokenizer::new(&[4]);
        let mut out = Vec::new();

        let generated =
            generate_with_writer(&mut model, &[1, 2], &tokenizer, 0.7, 6, Some(&mut out)).unwrap();

        assert_eq!(generated, vec![3, 5, 4]);
        assert_eq!(String::from_utf8(out).unwrap(), "AB!");
        let seq_positions: Vec<usize> = model.calls.iter().map(|(_, pos, _)| *pos).collect();
        assert_eq!(seq_positions, vec![0, 2, 3]);

        // Edge case 1: max_tokens cutoff before reaching a stop token (must not make an extra call).
        let mut model_cap = ToyModel::new(vec![3, 5, 4], 6);
        let mut out_cap = Vec::new();
        let gen_cap =
            generate_with_writer(&mut model_cap, &[1, 2], &tokenizer, 0.7, 2, Some(&mut out_cap))
                .unwrap();
        assert_eq!(gen_cap, vec![3, 5]);
        assert_eq!(String::from_utf8(out_cap).unwrap(), "AB");
        assert_eq!(model_cap.calls.len(), 2);

        // Edge case 2: max_tokens == 0 produces empty output and makes zero model calls.
        let mut model_zero = ToyModel::new(vec![3], 6);
        let gen_zero = generate_with_writer(&mut model_zero, &[1, 2], &tokenizer, 0.7, 0, None)
            .unwrap();
        assert!(gen_zero.is_empty());
        assert!(model_zero.calls.is_empty());
    }
}
