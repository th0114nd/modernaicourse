//! Grader-equivalent submission tests for Homework 6.
//!
//! Ported from `homework/hw6/hw6_tests.py` (`submit_*` functions) with verified
//! ground-truth assertions plus extra edge cases (weight save/load round-trip,
//! `deep_clone` independence, softplus stability & gradient at 0, masked CE edge cases,
//! DPO gradient isolation & beta scaling, and synthetic SFT + DPO convergence).
//! Run with: `cargo test -p hw6 --test submit`
//!
//! Full HuggingFace checkpoint evaluation tests (`submit_eval_llm_chat`,
//! `submit_eval_llm_dpo`) are marked `#[ignore]` so the default suite runs fast and
//! 100% offline.

mod submit {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use candle_core::{DType, Device, Result, Tensor, Var, backprop::GradStore};
    use common::{assert_allclose, seeded_randn};
    use hw6::{
        ASSISTANT_CLOSE, ASSISTANT_CLOSE_ID, ASSISTANT_OPEN, ASSISTANT_OPEN_ID, Adam, CausalLM,
        ChatTokenizer, DataLoaderChat, EOT_ID, Embedding, LLM, Linear, Message, ModelConfig,
        Optimizer, Tokenizer, USER_CLOSE, USER_CLOSE_ID, USER_OPEN, USER_OPEN_ID,
        convert_to_chat_format, cross_entropy_loss, download_hw6_files, dpo_loss,
        generate_with_writer, get_loss_mask, load_pretrained_llm, log_probs, mask_indices,
        pretokenize_chat, softplus, train_dpo, train_llm_chat,
    };

    static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(prefix: &str) -> Self {
            let id = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "hw6_submit_{prefix}_{}_{id}",
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

    /// Independent f64 reference for `log_probs`.
    fn ref_log_probs_f64(logits: &Tensor, y: &Tensor, mask: &Tensor) -> Tensor {
        let (b, s, v) = logits.dims3().unwrap();
        let z = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let yv = y.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        let mv = mask.flatten_all().unwrap().to_vec1::<u8>().unwrap();

        let mut out = vec![0.0_f32; b];
        for (bi, slot) in out.iter_mut().enumerate() {
            let mut sum = 0.0_f64;
            for si in 0..s {
                let idx = bi * s + si;
                if mv[idx] == 0 {
                    continue;
                }
                let row = &z[idx * v..(idx + 1) * v];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                let sum_exp: f64 = row.iter().map(|&x| ((x as f64) - max).exp()).sum();
                let lse = max + sum_exp.ln();
                let target = row[yv[idx] as usize] as f64;
                sum += target - lse;
            }
            *slot = sum as f32;
        }
        Tensor::from_vec(out, (b,), logits.device()).unwrap()
    }

    /// Independent f64 reference for `softplus(x, beta) = log(1 + exp(beta * x))`.
    fn ref_softplus_f64(x: &Tensor, beta: f64) -> Tensor {
        let vals = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let out: Vec<f32> = vals
            .into_iter()
            .map(|v| {
                let z = (v as f64) * beta;
                let res = if z > 0.0 {
                    z + (-z).exp().ln_1p()
                } else {
                    z.exp().ln_1p()
                };
                res as f32
            })
            .collect();
        Tensor::from_vec(out, x.shape(), x.device()).unwrap()
    }

    /// Independent f64 reference for `dpo_loss` on lookup tables.
    #[allow(clippy::too_many_arguments)]
    fn ref_dpo_loss_f64(
        model: &mut LookupLM,
        model_ref: &mut LookupLM,
        xp: &Tensor,
        yp: &Tensor,
        maskp: &Tensor,
        xn: &Tensor,
        yn: &Tensor,
        maskn: &Tensor,
        beta: f64,
    ) -> Tensor {
        let lp_p = ref_log_probs_f64(&model.forward(xp, 0, false).unwrap(), yp, maskp);
        let lp_n = ref_log_probs_f64(&model.forward(xn, 0, false).unwrap(), yn, maskn);
        let rp_p = ref_log_probs_f64(&model_ref.forward(xp, 0, false).unwrap(), yp, maskp);
        let rp_n = ref_log_probs_f64(&model_ref.forward(xn, 0, false).unwrap(), yn, maskn);
        let diff = ((lp_n - rp_n).unwrap() - (lp_p - rp_p).unwrap()).unwrap();
        ref_softplus_f64(&diff, beta)
    }

    // ---------------- Problem 1: convert_to_chat_format ----------------

    #[test]
    fn submit_convert_to_chat_format() {
        // Exact 3 cases from hw6_tests.py `submit_convert_to_chat_format`
        let case1 = vec![
            Message::user("What time is it?"),
            Message::assistant("It is noon."),
            Message::user("Thanks."),
            Message::assistant("You're welcome."),
        ];
        assert_eq!(
            convert_to_chat_format(&case1),
            "<USER>What time is it?</USER><ASSISTANT>It is noon.</ASSISTANT><USER>Thanks.</USER><ASSISTANT>You're welcome.</ASSISTANT>"
        );

        let case2 = vec![
            Message::assistant("System ready."),
            Message::assistant("Awaiting prompt."),
        ];
        assert_eq!(
            convert_to_chat_format(&case2),
            "<ASSISTANT>System ready.</ASSISTANT><ASSISTANT>Awaiting prompt.</ASSISTANT>"
        );

        let case3 = vec![
            Message::user("Line one.\nLine two."),
            Message::assistant("Tabbed\tresponse."),
            Message::user(""),
        ];
        assert_eq!(
            convert_to_chat_format(&case3),
            "<USER>Line one.\nLine two.</USER><ASSISTANT>Tabbed\tresponse.</ASSISTANT><USER></USER>"
        );

        // Extra edge case: empty conversation
        assert_eq!(convert_to_chat_format(&[]), "");
    }

    // ---------------- Problem 2: pretokenize_chat ----------------

    #[test]
    fn submit_pretokenize_chat() {
        // Exact 3 cases from hw6_tests.py `submit_pretokenize_chat`
        type PretokCase<'a> = (Vec<Vec<Message>>, Vec<(&'a str, &'a [u32])>, Vec<Vec<u32>>);
        let cases: Vec<PretokCase<'_>> = vec![
            (
                vec![
                    vec![Message::user("A"), Message::assistant("B")],
                    vec![Message::assistant("C")],
                ],
                vec![
                    ("<USER>A</USER><ASSISTANT>B</ASSISTANT>", &[1, 2, 3]),
                    ("<ASSISTANT>C</ASSISTANT>", &[4, 5]),
                ],
                vec![vec![1, 2, 3], vec![4, 5]],
            ),
            (
                vec![
                    vec![Message::user("First")],
                    vec![Message::user("Second"), Message::assistant("Third")],
                ],
                vec![
                    ("<USER>First</USER>", &[10, 11]),
                    (
                        "<USER>Second</USER><ASSISTANT>Third</ASSISTANT>",
                        &[12, 13, 14, 15],
                    ),
                ],
                vec![vec![10, 11], vec![12, 13, 14, 15]],
            ),
            (
                vec![
                    vec![Message::assistant("Alpha"), Message::user("Beta")],
                    vec![Message::user(""), Message::assistant("Gamma")],
                ],
                vec![
                    ("<ASSISTANT>Alpha</ASSISTANT><USER>Beta</USER>", &[21, 22, 23]),
                    ("<USER></USER><ASSISTANT>Gamma</ASSISTANT>", &[24, 25]),
                ],
                vec![vec![21, 22, 23], vec![24, 25]],
            ),
        ];

        for (i, (chats, mapping, expected)) in cases.into_iter().enumerate() {
            let tokenizer = FixedEncodeTokenizer::new(&mapping);
            let tmp = TempDir::new(&format!("pretok_{i}"));
            let in_path = tmp.join("chats.json");
            let out_path = tmp.join("tokens.json");
            fs::write(&in_path, serde_json::to_string(&chats).unwrap()).unwrap();

            pretokenize_chat(&tokenizer, &in_path, &out_path).unwrap();
            let got: Vec<Vec<u32>> =
                serde_json::from_str(&fs::read_to_string(&out_path).unwrap()).unwrap();
            assert_eq!(got, expected);
        }

        // Extra verification with real `ChatTokenizer` special token IDs matching `hw6.py`
        let real_tok = ChatTokenizer::new().unwrap();
        assert_eq!(real_tok.special_token("<|endoftext|>"), Some(EOT_ID));
        assert_eq!(real_tok.special_token(USER_OPEN), Some(USER_OPEN_ID));
        assert_eq!(real_tok.special_token(USER_CLOSE), Some(USER_CLOSE_ID));
        assert_eq!(
            real_tok.special_token(ASSISTANT_OPEN),
            Some(ASSISTANT_OPEN_ID)
        );
        assert_eq!(
            real_tok.special_token(ASSISTANT_CLOSE),
            Some(ASSISTANT_CLOSE_ID)
        );

        let chats = vec![vec![Message::user("Hello"), Message::assistant("World!")]];
        let tmp = TempDir::new("pretok_real");
        let in_path = tmp.join("chats.json");
        let out_path = tmp.join("tokens.json");
        fs::write(&in_path, serde_json::to_string(&chats).unwrap()).unwrap();
        pretokenize_chat(&real_tok, &in_path, &out_path).unwrap();
        let toks: Vec<Vec<u32>> =
            serde_json::from_str(&fs::read_to_string(&out_path).unwrap()).unwrap();
        assert_eq!(toks.len(), 1);
        assert_eq!(toks[0][0], USER_OPEN_ID);
        assert!(toks[0].contains(&USER_CLOSE_ID));
        assert!(toks[0].contains(&ASSISTANT_OPEN_ID));
        assert_eq!(*toks[0].last().unwrap(), ASSISTANT_CLOSE_ID);
        assert_eq!(
            real_tok.decode(&toks[0]).unwrap(),
            "<USER>Hello</USER><ASSISTANT>World!</ASSISTANT>"
        );
    }

    // ---------------- Problem 3: get_loss_mask ----------------

    #[test]
    fn submit_get_loss_mask() {
        let tokenizer = FakeChatTokenizer;
        // Exact 3 cases from hw6_tests.py `submit_get_loss_mask`
        let c1 = [91, 10, 93, 20, 21, 94, 93, 30, 94];
        assert_eq!(
            get_loss_mask(&c1, &tokenizer),
            vec![false, false, false, true, true, true, false, true, true]
        );

        let c2 = [93, 31, 32, 94, 9];
        assert_eq!(
            get_loss_mask(&c2, &tokenizer),
            vec![false, true, true, true, false]
        );

        let c3 = [91, 7, 8, 92, 9];
        assert_eq!(
            get_loss_mask(&c3, &tokenizer),
            vec![false, false, false, false, false]
        );

        // Extra edge cases: empty, unclosed assistant turn, orphan close tag, empty assistant turn
        assert_eq!(get_loss_mask(&[], &tokenizer), Vec::<bool>::new());
        assert_eq!(
            get_loss_mask(&[93, 1, 2], &tokenizer),
            vec![false, true, true]
        );
        assert_eq!(
            get_loss_mask(&[94, 1, 93, 94], &tokenizer),
            vec![false, false, false, true]
        );

        // Real ChatTokenizer multi-turn verification
        let real_tok = ChatTokenizer::new().unwrap();
        let text = convert_to_chat_format(&[
            Message::user("Q1"),
            Message::assistant("A1"),
            Message::user("Q2"),
            Message::assistant("A2"),
        ]);
        let ids = real_tok.encode(&text);
        let mask = get_loss_mask(&ids, &real_tok);
        let trained_tokens: Vec<u32> = ids
            .iter()
            .zip(mask.iter())
            .filter_map(|(&id, &keep)| keep.then_some(id))
            .collect();
        assert_eq!(
            real_tok.decode(&trained_tokens).unwrap(),
            "A1</ASSISTANT>A2</ASSISTANT>"
        );
    }

    // ---------------- Problem 4: DataLoaderChat ----------------

    #[test]
    fn submit_dataloader_chat() {
        let tokenizer = FakeChatTokenizer;

        // Exact 3 cases from hw6_tests.py `submit_DataLoaderChat`
        // Case 1
        {
            let chats = vec![vec![1u32, 93, 10, 11, 94], vec![2, 93, 12, 94]];
            let tmp = TempDir::new("dl1");
            let path = tmp.join("t.json");
            fs::write(&path, serde_json::to_string(&chats).unwrap()).unwrap();
            let loader = DataLoaderChat::new(&path, 5, 2, &tokenizer, Device::Cpu).unwrap();
            assert_eq!(loader.len(), 1);
            let batches: Vec<_> = loader.iter().map(|b| b.unwrap()).collect();
            let (x, y, m) = &batches[0];
            assert_eq!(
                x.to_vec2::<u32>().unwrap(),
                vec![vec![1, 93, 10, 11, 94], vec![2, 93, 12, 94, 0]]
            );
            assert_eq!(
                y.to_vec2::<u32>().unwrap(),
                vec![vec![93, 10, 11, 94, 0], vec![93, 12, 94, 0, 0]]
            );
            assert_eq!(
                m.to_vec2::<u8>().unwrap(),
                vec![vec![0, 1, 1, 1, 0], vec![0, 1, 1, 0, 0]]
            );
        }

        // Case 2
        {
            let chats = vec![
                vec![3u32, 93, 13, 14, 94],
                vec![4, 93, 15, 94],
                vec![5, 93, 16, 17, 94],
                vec![6, 93, 18, 94],
            ];
            let tmp = TempDir::new("dl2");
            let path = tmp.join("t.json");
            fs::write(&path, serde_json::to_string(&chats).unwrap()).unwrap();
            let loader = DataLoaderChat::new(&path, 5, 2, &tokenizer, Device::Cpu).unwrap();
            assert_eq!(loader.len(), 2);
            let batches: Vec<_> = loader.iter().map(|b| b.unwrap()).collect();
            let (x, y, m) = &batches[1];
            assert_eq!(
                x.to_vec2::<u32>().unwrap(),
                vec![vec![5, 93, 16, 17, 94], vec![6, 93, 18, 94, 0]]
            );
            assert_eq!(
                y.to_vec2::<u32>().unwrap(),
                vec![vec![93, 16, 17, 94, 0], vec![93, 18, 94, 0, 0]]
            );
            assert_eq!(
                m.to_vec2::<u8>().unwrap(),
                vec![vec![0, 1, 1, 1, 0], vec![0, 1, 1, 0, 0]]
            );
        }

        // Case 3
        {
            let chats = vec![vec![7u32, 93, 19, 20, 21, 94]];
            let tmp = TempDir::new("dl3");
            let path = tmp.join("t.json");
            fs::write(&path, serde_json::to_string(&chats).unwrap()).unwrap();
            let loader = DataLoaderChat::new(&path, 6, 1, &tokenizer, Device::Cpu).unwrap();
            assert_eq!(loader.len(), 1);
            let batches: Vec<_> = loader.iter().map(|b| b.unwrap()).collect();
            let (x, y, m) = &batches[0];
            assert_eq!(
                x.to_vec2::<u32>().unwrap(),
                vec![vec![7, 93, 19, 20, 21, 94]]
            );
            assert_eq!(
                y.to_vec2::<u32>().unwrap(),
                vec![vec![93, 19, 20, 21, 94, 0]]
            );
            assert_eq!(m.to_vec2::<u8>().unwrap(), vec![vec![0, 1, 1, 1, 1, 0]]);
        }

        // Extra edge cases: conversation starting directly with <ASSISTANT>, truncation when
        // len > seq_len, and empty conversation.
        {
            let chats = vec![
                vec![93u32, 10, 11, 12, 13, 94], // starts at <ASSISTANT>, len 6 > seq_len 4
                vec![],                          // empty chat
            ];
            let tmp = TempDir::new("dl_edge");
            let path = tmp.join("t.json");
            fs::write(&path, serde_json::to_string(&chats).unwrap()).unwrap();
            let loader = DataLoaderChat::new(&path, 4, 2, &tokenizer, Device::Cpu).unwrap();
            let (x, y, m) = loader.iter().next().unwrap().unwrap();
            assert_eq!(
                x.to_vec2::<u32>().unwrap(),
                vec![vec![93, 10, 11, 12], vec![0, 0, 0, 0]]
            );
            assert_eq!(
                y.to_vec2::<u32>().unwrap(),
                vec![vec![10, 11, 12, 13], vec![0, 0, 0, 0]]
            );
            assert_eq!(
                m.to_vec2::<u8>().unwrap(),
                vec![vec![1, 1, 1, 1], vec![0, 0, 0, 0]]
            );
            assert!(DataLoaderChat::new(&path, 0, 2, &tokenizer, Device::Cpu).is_err());
            assert!(DataLoaderChat::new(&path, 4, 0, &tokenizer, Device::Cpu).is_err());
        }
    }

    // ---------------- Problem 5: train_llm_chat ----------------

    #[test]
    fn submit_train_llm_chat() {
        // Exact 3 training cases from hw6_tests.py `submit_train_llm_chat`, verified against
        // an independent masked-CE + SGD reference pass.
        type RawBatch = (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<Vec<bool>>);
        type TrainChatCase = (u64, Vec<RawBatch>, Vec<u32>, Option<usize>);
        let cases: Vec<TrainChatCase> = vec![
            (
                1,
                vec![(
                    vec![vec![0, 1, 2], vec![1, 2, 3]],
                    vec![vec![1, 2, 3], vec![2, 3, 4]],
                    vec![vec![false, true, true], vec![true, false, true]],
                )],
                vec![0, 1, 2],
                Some(1),
            ),
            (
                2,
                vec![
                    (
                        vec![vec![0, 1, 2], vec![1, 2, 3]],
                        vec![vec![1, 2, 3], vec![2, 3, 4]],
                        vec![vec![false, true, true], vec![true, false, true]],
                    ),
                    (
                        vec![vec![2, 3, 4], vec![0, 2, 4]],
                        vec![vec![3, 4, 0], vec![2, 4, 1]],
                        vec![vec![true, true, false], vec![false, true, true]],
                    ),
                ],
                vec![2, 3, 4],
                None,
            ),
            (
                3,
                vec![
                    (
                        vec![vec![4, 3, 2], vec![1, 0, 5]],
                        vec![vec![3, 2, 1], vec![0, 5, 4]],
                        vec![vec![true, false, true], vec![false, true, true]],
                    ),
                    (
                        vec![vec![5, 4, 3], vec![2, 1, 0]],
                        vec![vec![4, 3, 2], vec![1, 0, 5]],
                        vec![vec![true, true, false], vec![true, false, true]],
                    ),
                ],
                vec![1, 0, 5],
                Some(2),
            ),
        ];

        for (seed, raw_batches, eval_ids, max_iter) in cases {
            let build_batches = || -> Vec<Result<(Tensor, Tensor, Tensor)>> {
                raw_batches
                    .iter()
                    .map(|(x, y, m)| {
                        let xr: Vec<&[u32]> = x.iter().map(Vec::as_slice).collect();
                        let yr: Vec<&[u32]> = y.iter().map(Vec::as_slice).collect();
                        let mr: Vec<&[bool]> = m.iter().map(Vec::as_slice).collect();
                        Ok((u32_2d(&xr), u32_2d(&yr), mask_2d(&mr)))
                    })
                    .collect()
            };

            let mut model = TinyChatTrainModel::new_seeded(6, 4, seed);
            let mut ref_model = model.clone_fresh();

            let mut opt = SimpleSgd::new(model.parameters(), 0.1);
            train_llm_chat(&mut model, build_batches(), &mut opt, max_iter).unwrap();

            let mut ref_opt = SimpleSgd::new(ref_model.parameters(), 0.1);
            let limit = max_iter.unwrap_or(raw_batches.len());
            for batch in build_batches().into_iter().take(limit) {
                let (x, y, m) = batch.unwrap();
                let logits = ref_model.forward(&x, 0, false).unwrap();
                let (b, s, v) = logits.dims3().unwrap();
                let idx = mask_indices(&m).unwrap();
                let loss = cross_entropy_loss(
                    &logits
                        .reshape((b * s, v))
                        .unwrap()
                        .index_select(&idx, 0)
                        .unwrap(),
                    &y.reshape((b * s,)).unwrap().index_select(&idx, 0).unwrap(),
                )
                .unwrap();
                ref_opt.step(&loss.backward().unwrap()).unwrap();
            }

            let eval_t = Tensor::from_slice(&eval_ids, (1, eval_ids.len()), &Device::Cpu).unwrap();
            let got_logits = model.forward(&eval_t, 0, false).unwrap();
            let ref_logits = ref_model.forward(&eval_t, 0, false).unwrap();
            assert_allclose(&got_logits, &ref_logits, 1e-5, 1e-5);
        }

        // Extra edge case 1: unmasked target positions must have zero effect on updates.
        {
            let mut m_a = TinyChatTrainModel::new_seeded(6, 4, 42);
            let mut m_b = m_a.clone_fresh();
            let mut opt_a = SimpleSgd::new(m_a.parameters(), 0.2);
            let mut opt_b = SimpleSgd::new(m_b.parameters(), 0.2);
            let b_a = vec![Ok((
                u32_2d(&[&[0, 1, 2]]),
                u32_2d(&[&[5, 2, 3]]), // pos 0 is masked out (target 5)
                mask_2d(&[&[false, true, true]]),
            ))];
            let b_b = vec![Ok((
                u32_2d(&[&[0, 1, 2]]),
                u32_2d(&[&[0, 2, 3]]), // pos 0 target changed to 0 -> must not matter
                mask_2d(&[&[false, true, true]]),
            ))];
            train_llm_chat(&mut m_a, b_a, &mut opt_a, None).unwrap();
            train_llm_chat(&mut m_b, b_b, &mut opt_b, None).unwrap();
            assert_allclose(
                m_a.embed.weight.as_tensor(),
                m_b.embed.weight.as_tensor(),
                1e-6,
                1e-6,
            );
            assert_allclose(
                m_a.proj.weight.as_tensor(),
                m_b.proj.weight.as_tensor(),
                1e-6,
                1e-6,
            );
        }

        // Extra edge case 2: max_iter = Some(0) and all-false mask batch must not update or NaN.
        {
            let mut model = TinyChatTrainModel::new_seeded(6, 4, 77);
            let init_w = model.proj.weight.as_detached_tensor().copy().unwrap();
            let mut opt = SimpleSgd::new(model.parameters(), 0.1);
            let batch = vec![Ok((
                u32_2d(&[&[0, 1, 2]]),
                u32_2d(&[&[1, 2, 3]]),
                mask_2d(&[&[true, true, true]]),
            ))];
            train_llm_chat(&mut model, batch, &mut opt, Some(0)).unwrap();
            assert_eq!(opt.steps, 0);
            assert_allclose(model.proj.weight.as_tensor(), &init_w, 0.0, 0.0);

            let empty_mask_batch = vec![Ok((
                u32_2d(&[&[0, 1, 2]]),
                u32_2d(&[&[1, 2, 3]]),
                mask_2d(&[&[false, false, false]]),
            ))];
            train_llm_chat(&mut model, empty_mask_batch, &mut opt, None).unwrap();
            assert_allclose(model.proj.weight.as_tensor(), &init_w, 0.0, 0.0);
        }

        // Extra edge case 3: synthetic SFT convergence on a small LLM with Adam.
        {
            let device = Device::Cpu;
            let mut model = LLM::new(16, 16, 2, 8, 32, 1, &device).unwrap();
            let x = u32_2d(&[&[1, 2, 3, 4, 5, 6], &[1, 7, 3, 8, 9, 6]]);
            let y = u32_2d(&[&[2, 3, 4, 5, 6, 0], &[7, 3, 8, 9, 6, 0]]);
            let mask = mask_2d(&[
                &[false, false, true, true, true, false],
                &[false, false, true, true, true, false],
            ]);
            let eval_loss = |m: &mut LLM| -> f32 {
                let logits = m.forward(&x, 0, false).unwrap();
                let idx = mask_indices(&mask).unwrap();
                cross_entropy_loss(
                    &logits.reshape((12, 16)).unwrap().index_select(&idx, 0).unwrap(),
                    &y.reshape((12,)).unwrap().index_select(&idx, 0).unwrap(),
                )
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
            };
            let loss_before = eval_loss(&mut model);
            let mut opt = Adam::new(model.parameters(), 2e-2, (0.9, 0.95), 1e-8).unwrap();
            let batches = (0..25).map(|_| Ok((x.clone(), y.clone(), mask.clone())));
            train_llm_chat(&mut model, batches, &mut opt, None).unwrap();
            let loss_after = eval_loss(&mut model);
            assert!(
                loss_after < loss_before * 0.25,
                "expected SFT loss to drop sharply: before={loss_before}, after={loss_after}"
            );
        }
    }

    // ---------------- Problem 6: log_probs ----------------

    #[test]
    fn submit_log_probs() {
        let device = Device::Cpu;

        // Exact 3 cases from hw6_tests.py `submit_log_probs`
        let cases = vec![
            (
                Tensor::from_vec(
                    vec![
                        1.0_f32, 0.0, 0.0, 1.0, 2.0, -1.0, // b0
                        -0.5, 0.5, 1.5, 1.5, 0.25, -0.25, // b1
                    ],
                    (2, 3, 2),
                    &device,
                )
                .unwrap(),
                u32_2d(&[&[0, 1, 0], &[1, 0, 1]]),
                mask_2d(&[&[true, true, false], &[false, true, true]]),
            ),
            (
                Tensor::from_vec(
                    vec![
                        2.0_f32, 1.0, 0.0, 0.0, 0.0, 0.0, // b0
                        -1.0, 2.0, 1.0, 3.0, 0.0, -3.0, // b1
                    ],
                    (2, 2, 3),
                    &device,
                )
                .unwrap(),
                u32_2d(&[&[1, 2], &[0, 0]]),
                mask_2d(&[&[true, false], &[true, true]]),
            ),
            (
                Tensor::from_vec(vec![0.5_f32, -0.5, 1.0, -1.0, 0.0, 0.0], (1, 3, 2), &device)
                    .unwrap(),
                u32_2d(&[&[0, 0, 1]]),
                mask_2d(&[&[false, true, true]]),
            ),
        ];

        for (logits, y, mask) in cases {
            let got = log_probs(&logits, &y, &mask).unwrap();
            let expected = ref_log_probs_f64(&logits, &y, &mask);
            assert_eq!(got.dims(), expected.dims());
            assert_allclose(&got, &expected, 1e-5, 1e-5);
        }

        // Extra edge cases: all-false mask row (= 0.0) & large shifted logits (+1000)
        let big = Tensor::from_vec(
            vec![1002.0_f32, 1000.0, 999.0, -1000.0, -1001.0, -999.0],
            (2, 1, 3),
            &device,
        )
        .unwrap();
        let y_big = u32_2d(&[&[0], &[2]]);
        let m_big = mask_2d(&[&[true], &[false]]);
        let got_big = log_probs(&big, &y_big, &m_big).unwrap();
        let exp_big = ref_log_probs_f64(&big, &y_big, &m_big);
        assert_allclose(&got_big, &exp_big, 1e-4, 1e-4);
        assert_eq!(got_big.to_vec1::<f32>().unwrap()[1], 0.0);

        // Gradient check via finite differences
        let v = Var::from_vec(vec![1.2_f32, -0.3, 0.7, 0.1, 0.9, -0.4], (1, 2, 3), &device)
            .unwrap();
        let y_g = u32_2d(&[&[2, 0]]);
        let m_g = mask_2d(&[&[true, false]]);
        let lp = log_probs(v.as_tensor(), &y_g, &m_g).unwrap();
        let grads = lp.sum_all().unwrap().backward().unwrap();
        let g_vec = grads
            .get(v.as_tensor())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // Position 1 is masked out -> zero gradient; position 0 gradient is (one_hot(2) - softmax).
        assert_eq!(g_vec[3], 0.0);
        assert_eq!(g_vec[4], 0.0);
        assert_eq!(g_vec[5], 0.0);
        let row0 = [1.2_f64, -0.3, 0.7];
        let max0 = 1.2_f64;
        let exps: Vec<f64> = row0.iter().map(|&z| (z - max0).exp()).collect();
        let sum_e: f64 = exps.iter().sum();
        for j in 0..3 {
            let expected_g = (if j == 2 { 1.0 } else { 0.0 }) - exps[j] / sum_e;
            assert!(
                (g_vec[j] as f64 - expected_g).abs() < 1e-5,
                "grad[{j}] mismatch: got {}, expected {expected_g}",
                g_vec[j]
            );
        }
    }

    // ---------------- Problem 7a: softplus ----------------

    #[test]
    fn submit_softplus() {
        let device = Device::Cpu;
        // Exact 3 cases from hw6_tests.py `submit_softplus`
        let cases = vec![
            (
                Tensor::from_slice(&[-2.0_f32, 0.0, 1.5, 3.0], (4,), &device).unwrap(),
                0.4,
            ),
            (
                Tensor::from_vec(vec![1.0_f32, -1.0, 2.5, -2.5], (2, 2), &device).unwrap(),
                0.8,
            ),
            (Tensor::from_slice(&[0.25_f32], (1,), &device).unwrap(), 0.2),
        ];
        for (x, beta) in cases {
            let got = softplus(&x, beta).unwrap();
            let expected = ref_softplus_f64(&x, beta);
            assert_allclose(&got, &expected, 1e-5, 1e-5);
        }

        // Extreme values stability & exact gradient (including at x = 0.0!)
        let x_ext = Var::from_slice(&[-1000.0_f32, -2.0, 0.0, 2.0, 1000.0], (5,), &device)
            .unwrap();
        let beta = 0.5;
        let out = softplus(x_ext.as_tensor(), beta).unwrap();
        let expected = ref_softplus_f64(x_ext.as_tensor(), beta);
        assert_allclose(&out, &expected, 1e-4, 1e-4);

        let grads = out.sum_all().unwrap().backward().unwrap();
        let g = grads
            .get(x_ext.as_tensor())
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected_g: Vec<f32> = [-1000.0_f64, -2.0, 0.0, 2.0, 1000.0]
            .iter()
            .map(|&v| {
                let z = beta * v;
                let sig = if z >= 0.0 {
                    1.0 / (1.0 + (-z).exp())
                } else {
                    let ez = z.exp();
                    ez / (1.0 + ez)
                };
                (beta * sig) as f32
            })
            .collect();
        for (i, (&gi, &ei)) in g.iter().zip(expected_g.iter()).enumerate() {
            assert!(
                gi.is_finite() && (gi - ei).abs() < 1e-5,
                "softplus grad[{i}] mismatch: got {gi}, expected {ei}"
            );
        }
    }

    // ---------------- Problem 7b: dpo_loss ----------------

    #[test]
    fn submit_dpo_loss() {
        // Exact 3 cases from hw6_tests.py `submit_dpo_loss`
        let cases = vec![
            (
                vec![1.5_f32, 0.0, -0.5, 0.5, 1.5, -1.0, 0.25, -0.75, 1.25],
                vec![1.0_f32, 0.0, -0.25, 0.0, 1.0, -0.5, 0.0, -0.5, 1.0],
                3,
                3,
                u32_2d(&[&[0, 1, 2]]),
                u32_2d(&[&[1, 2, 0]]),
                mask_2d(&[&[true, true, false]]),
                u32_2d(&[&[2, 1, 0]]),
                u32_2d(&[&[2, 0, 1]]),
                mask_2d(&[&[true, false, true]]),
                0.2,
            ),
            (
                vec![2.0_f32, -1.0, 0.5, 1.0],
                vec![1.0_f32, -0.5, 0.0, 0.75],
                2,
                2,
                u32_2d(&[&[0, 1], &[1, 0]]),
                u32_2d(&[&[1, 0], &[0, 1]]),
                mask_2d(&[&[true, false], &[true, true]]),
                u32_2d(&[&[1, 0], &[0, 1]]),
                u32_2d(&[&[0, 1], &[1, 0]]),
                mask_2d(&[&[true, true], &[false, true]]),
                0.5,
            ),
            (
                vec![
                    1.0_f32, 0.0, -1.0, 0.5, 0.25, 1.5, -0.5, 0.0, 1.25, -0.25, 0.5, -1.0, 0.75,
                    0.25, -0.75, 1.0,
                ],
                vec![
                    0.5_f32, 0.0, -0.5, 0.25, 0.0, 1.0, -0.25, 0.0, 1.0, -0.5, 0.25, -0.5, 0.5,
                    0.0, -0.25, 0.75,
                ],
                4,
                4,
                u32_2d(&[&[0, 1, 2, 3]]),
                u32_2d(&[&[1, 2, 3, 0]]),
                mask_2d(&[&[false, true, true, false]]),
                u32_2d(&[&[3, 2, 1, 0]]),
                u32_2d(&[&[2, 1, 0, 3]]),
                mask_2d(&[&[true, false, true, true]]),
                0.1,
            ),
        ];

        for (t, rt, rows, vocab, xp, yp, mp, xn, yn, mn, beta) in cases {
            let mut m = LookupLM::new(t, rows, vocab);
            let mut mr = LookupLM::new(rt, rows, vocab);
            let got = dpo_loss(&mut m, &mut mr, &xp, &yp, &mp, &xn, &yn, &mn, beta).unwrap();
            let expected = ref_dpo_loss_f64(&mut m, &mut mr, &xp, &yp, &mp, &xn, &yn, &mn, beta);
            assert_allclose(&got, &expected, 1e-5, 1e-5);

            // Gradient isolation check on every case
            let grads = got.sum_all().unwrap().backward().unwrap();
            assert!(grads.get(m.table.as_tensor()).is_some());
            assert!(grads.get(mr.table.as_tensor()).is_none());
        }

        // Extra edge cases:
        // 1. When model == model_ref, DPO loss must equal ln(2) for every beta.
        let mut m_same = LookupLM::new(vec![1.0, -0.5, 0.2, 0.3, 0.8, -1.1], 2, 3);
        let mut mr_same = m_same.clone_fresh();
        let xp = u32_2d(&[&[0, 1]]);
        let yp = u32_2d(&[&[1, 2]]);
        let mp = mask_2d(&[&[true, true]]);
        let xn = u32_2d(&[&[1, 0]]);
        let yn = u32_2d(&[&[0, 1]]);
        let mn = mask_2d(&[&[true, true]]);
        for beta in [0.05, 0.1, 0.5, 2.0] {
            let loss = dpo_loss(&mut m_same, &mut mr_same, &xp, &yp, &mp, &xn, &yn, &mn, beta)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()[0];
            assert!(
                (loss - 2.0_f32.ln()).abs() < 1e-5,
                "expected ln(2) when model == model_ref, got {loss} for beta={beta}"
            );
        }

        // 2. Beta scaling monotonicity when policy favors positive over negative vs ref.
        let mut m_good = LookupLM::new(vec![0.0, 3.0, 0.0, 0.0, 0.0, 3.0], 2, 3);
        let mut mr_flat = LookupLM::new(vec![0.0; 6], 2, 3);
        let l_small_beta =
            dpo_loss(&mut m_good, &mut mr_flat, &xp, &yp, &mp, &xn, &yn, &mn, 0.1)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()[0];
        let l_large_beta =
            dpo_loss(&mut m_good, &mut mr_flat, &xp, &yp, &mp, &xn, &yn, &mn, 1.0)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()[0];
        assert!(l_large_beta < l_small_beta && l_small_beta < 2.0_f32.ln());
    }

    // ---------------- Problem 8: train_dpo ----------------

    #[test]
    fn submit_train_dpo() {
        // Exact multi-case schedule modeled on hw6_tests.py `submit_train_dpo`:
        // (max_iter=Some(1), max_iter=None, max_iter=Some(2) on 3-batch loader),
        // verified against independent step-by-step reference DPO + SGD updates.
        let base_pos = || -> Vec<Result<(Tensor, Tensor, Tensor)>> {
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
        let base_neg = || -> Vec<Result<(Tensor, Tensor, Tensor)>> {
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
        let extra_pos = || -> Vec<Result<(Tensor, Tensor, Tensor)>> {
            let mut v = base_pos();
            v.push(Ok((
                u32_2d(&[&[3, 1]]),
                u32_2d(&[&[1, 2]]),
                mask_2d(&[&[true, true]]),
            )));
            v
        };
        let extra_neg = || -> Vec<Result<(Tensor, Tensor, Tensor)>> {
            let mut v = base_neg();
            v.push(Ok((
                u32_2d(&[&[2, 0]]),
                u32_2d(&[&[2, 1]]),
                mask_2d(&[&[false, true]]),
            )));
            v
        };

        let init_table = vec![
            0.8_f32, -0.2, 0.1, // 0
            -0.3, 0.5, 0.4, // 1
            0.2, 0.7, -0.6, // 2
            -0.5, 0.1, 0.9, // 3
        ];

        for (use_extra, beta, lr, max_iter, expected_steps) in [
            (false, 0.2, 0.1, Some(1), 1usize),
            (false, 0.2, 0.1, None, 2usize),
            (true, 0.2, 0.05, Some(2), 2usize),
        ] {
            let mut model = LookupLM::new(init_table.clone(), 4, 3);
            let mut model_ref = model.clone_fresh();
            let mut ref_check_model = model.clone_fresh();
            let mut ref_check_ref = model.clone_fresh();
            let ref_snapshot = model_ref.table.as_detached_tensor().copy().unwrap();

            let mut opt = SimpleSgd::new(vec![model.table.clone()], lr);
            let pos = if use_extra { extra_pos() } else { base_pos() };
            let neg = if use_extra { extra_neg() } else { base_neg() };
            train_dpo(&mut model, &mut model_ref, pos, neg, &mut opt, beta, max_iter).unwrap();
            assert_eq!(opt.steps, expected_steps);
            assert_allclose(model_ref.table.as_tensor(), &ref_snapshot, 0.0, 0.0);

            // Manual reference updates
            let mut ref_opt = SimpleSgd::new(vec![ref_check_model.table.clone()], lr);
            let pos2 = if use_extra { extra_pos() } else { base_pos() };
            let neg2 = if use_extra { extra_neg() } else { base_neg() };
            for (pb, nb) in pos2.into_iter().zip(neg2).take(expected_steps) {
                let (xp, yp, mp) = pb.unwrap();
                let (xn, yn, mn) = nb.unwrap();
                let loss = dpo_loss(
                    &mut ref_check_model,
                    &mut ref_check_ref,
                    &xp,
                    &yp,
                    &mp,
                    &xn,
                    &yn,
                    &mn,
                    beta,
                )
                .unwrap()
                .mean_all()
                .unwrap();
                ref_opt.step(&loss.backward().unwrap()).unwrap();
            }
            assert_allclose(
                model.table.as_tensor(),
                ref_check_model.table.as_tensor(),
                1e-5,
                1e-5,
            );
        }

        // Synthetic end-to-end DPO convergence on a small LLM with `deep_clone` + `Adam`:
        // preferred completion log-prob margin must increase while `model_ref` remains frozen.
        {
            let device = Device::Cpu;
            let mut model = LLM::new(16, 16, 2, 8, 32, 1, &device).unwrap();
            let mut model_ref = model.deep_clone().unwrap();
            let ref_embed_before = model_ref.embedding.weight.as_detached_tensor().copy().unwrap();

            let xp = u32_2d(&[&[1, 2, 3, 4, 5]]);
            let yp = u32_2d(&[&[2, 3, 4, 5, 6]]);
            let mp = mask_2d(&[&[false, false, true, true, true]]);
            let xn = u32_2d(&[&[1, 2, 3, 9, 10]]);
            let yn = u32_2d(&[&[2, 3, 9, 10, 6]]);
            let mn = mask_2d(&[&[false, false, true, true, true]]);

            let loss_before = dpo_loss(&mut model, &mut model_ref, &xp, &yp, &mp, &xn, &yn, &mn, 0.2)
                .unwrap()
                .mean_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!((loss_before - 2.0_f32.ln()).abs() < 1e-4);

            let mut opt = Adam::new(model.parameters(), 1e-2, (0.9, 0.95), 1e-8).unwrap();
            let pos_iter = (0..20).map(|_| Ok((xp.clone(), yp.clone(), mp.clone())));
            let neg_iter = (0..20).map(|_| Ok((xn.clone(), yn.clone(), mn.clone())));
            train_dpo(&mut model, &mut model_ref, pos_iter, neg_iter, &mut opt, 0.2, None)
                .unwrap();

            let loss_after = dpo_loss(&mut model, &mut model_ref, &xp, &yp, &mp, &xn, &yn, &mn, 0.2)
                .unwrap()
                .mean_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(
                loss_after < 0.15,
                "expected DPO loss to decrease sharply from ln(2)={loss_before}, got {loss_after}"
            );
            assert_allclose(
                model_ref.embedding.weight.as_tensor(),
                &ref_embed_before,
                0.0,
                0.0,
            );
        }
    }

    // ---------------- Provided Scaffolding: LLM save/load round-trip & deep_clone ----------------

    #[test]
    fn submit_llm_weights_and_deep_clone() {
        let device = Device::Cpu;
        let mut orig = LLM::new(17, 12, 3, 8, 24, 2, &device).unwrap();
        let tokens = u32_2d(&[&[1, 2, 3, 4, 5]]);
        let orig_logits = orig.forward(&tokens, 0, false).unwrap();

        // 1. `.safetensors` save_weights -> load_weights exact round-trip
        let tmp = TempDir::new("weights_rt");
        let ckpt_path = tmp.join("model.safetensors");
        orig.save_weights(&ckpt_path).unwrap();

        let mut loaded = LLM::new(17, 12, 3, 8, 24, 2, &device).unwrap();
        loaded.load_weights(&ckpt_path).unwrap();
        let loaded_logits = loaded.forward(&tokens, 0, false).unwrap();
        assert_allclose(&loaded_logits, &orig_logits, 1e-6, 1e-6);

        // 2. BF16 save_weights_bf16 -> load_weights approximate round-trip
        let bf16_path = tmp.join("model_bf16.safetensors");
        orig.save_weights_bf16(&bf16_path).unwrap();
        let mut loaded_bf16 = LLM::new(17, 12, 3, 8, 24, 2, &device).unwrap();
        loaded_bf16.load_weights(&bf16_path).unwrap();
        let bf16_logits = loaded_bf16.forward(&tokens, 0, false).unwrap();
        assert_allclose(&bf16_logits, &orig_logits, 5e-2, 5e-2);

        // 3. `deep_clone` must allocate independent `Var`s with distinct `TensorId`s
        let mut cloned = orig.deep_clone().unwrap();
        for (p_orig, p_clone) in orig.parameters().iter().zip(cloned.parameters().iter()) {
            assert_ne!(
                p_orig.id(),
                p_clone.id(),
                "deep_clone must allocate fresh TensorIds"
            );
            assert_allclose(p_orig.as_tensor(), p_clone.as_tensor(), 0.0, 0.0);
        }

        // Mutate `orig` heavily and verify `cloned` logits are completely unaffected.
        let zeros = Tensor::zeros_like(orig.output.weight.as_tensor()).unwrap();
        orig.output.weight.set(&zeros).unwrap();
        let cloned_logits_after = cloned.forward(&tokens, 0, false).unwrap();
        assert_allclose(&cloned_logits_after, &orig_logits, 1e-6, 1e-6);
    }

    // ---------------- Updated Part 0l: generate with Option<u32> eot_token ----------------

    #[test]
    fn submit_generate() {
        struct SeqModel {
            next_tokens: Vec<u32>,
            calls: usize,
        }
        impl CausalLM for SeqModel {
            fn forward(
                &mut self,
                tokens: &Tensor,
                _seq_pos: usize,
                _use_kv_cache: bool,
            ) -> Result<Tensor> {
                let s = tokens.dim(1)?;
                let mut logits = vec![-1e9_f32; s * 8];
                let tok = self.next_tokens[self.calls] as usize;
                self.calls += 1;
                logits[(s - 1) * 8 + tok] = 0.0;
                Tensor::from_vec(logits, (1, s, 8), tokens.device())
            }
        }
        struct NumTok;
        impl Tokenizer for NumTok {
            fn encode(&self, _text: &str) -> Vec<u32> {
                Vec::new()
            }
            fn decode(&self, tokens: &[u32]) -> Result<String> {
                Ok(tokens.iter().map(|t| t.to_string()).collect())
            }
            fn special_token(&self, _name: &str) -> Option<u32> {
                None
            }
        }

        // eot_token = None runs all max_tokens without stopping early
        let mut m1 = SeqModel {
            next_tokens: vec![2, 4, 6, 7],
            calls: 0,
        };
        let mut out1 = Vec::new();
        let gen1 =
            generate_with_writer(&mut m1, &[1], &NumTok, None, 0.5, 4, Some(&mut out1)).unwrap();
        assert_eq!(gen1, vec![2, 4, 6, 7]);
        assert_eq!(String::from_utf8(out1).unwrap(), "2467");

        // eot_token = Some(4) stops immediately after token 4
        let mut m2 = SeqModel {
            next_tokens: vec![2, 4, 6, 7],
            calls: 0,
        };
        let gen2 = generate_with_writer(&mut m2, &[1], &NumTok, Some(4), 0.5, 4, None).unwrap();
        assert_eq!(gen2, vec![2, 4]);
        assert_eq!(m2.calls, 2);

        // max_tokens = 0 returns empty immediately
        let mut m3 = SeqModel {
            next_tokens: vec![2],
            calls: 0,
        };
        let gen3 = generate_with_writer(&mut m3, &[1], &NumTok, None, 0.5, 0, None).unwrap();
        assert!(gen3.is_empty());
        assert_eq!(m3.calls, 0);
    }

    // ---------------- Optional Full HF Checkpoint Submission Evals ----------------

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
        let avg_close = close_probs.iter().sum::<f64>() / close_probs.len() as f64;
        (total_loss / total_tokens as f64, avg_close)
    }

    #[test]
    #[ignore = "downloads model_base.pth and ultrachat_short.json from HuggingFace Hub and runs 10 SFT steps"]
    fn submit_eval_llm_chat() {
        let device = Device::Cpu;
        let files = download_hw6_files().unwrap();
        let tokenizer = ChatTokenizer::new().unwrap();
        let cfg = ModelConfig::from_file(&files.params).unwrap();

        let raw_chats: Vec<Vec<Message>> =
            serde_json::from_str(&fs::read_to_string(&files.ultrachat_short).unwrap()).unwrap();
        let tmp = TempDir::new("submit_eval_chat");
        let sub_in = tmp.join("sub.json");
        let sub_tok = tmp.join("sub_tok.json");
        fs::write(&sub_in, serde_json::to_string(&raw_chats[..30]).unwrap()).unwrap();
        pretokenize_chat(&tokenizer, &sub_in, &sub_tok).unwrap();

        let loader = DataLoaderChat::new(&sub_tok, cfg.max_seq_len, 2, &tokenizer, device.clone())
            .unwrap();
        let holdout: Vec<(Tensor, Tensor, Tensor)> =
            loader.iter().skip(12).take(1).map(|b| b.unwrap()).collect();

        let mut model = load_pretrained_llm(&files.params, &files.model_base, &device).unwrap();
        let mut base_model = model.deep_clone().unwrap();
        let mut opt = Adam::new(model.parameters(), 2e-5, (0.9, 0.95), 1e-8).unwrap();
        train_llm_chat(&mut model, &loader, &mut opt, Some(10)).unwrap();

        let (loss, close_prob) =
            masked_nll_and_close_prob(&mut model, &holdout, ASSISTANT_CLOSE_ID);
        let (base_loss, base_close_prob) =
            masked_nll_and_close_prob(&mut base_model, &holdout, ASSISTANT_CLOSE_ID);
        assert!(loss < base_loss);
        assert!(loss < 2.4);
        assert!(close_prob > (0.005_f64).max(1000.0 * base_close_prob));
    }

    #[test]
    #[ignore = "downloads model_chat.pth and UltraChat DPO datasets from HuggingFace Hub and runs 10 DPO steps"]
    fn submit_eval_llm_dpo() {
        let device = Device::Cpu;
        let files = download_hw6_files().unwrap();
        let tokenizer = ChatTokenizer::new().unwrap();
        let cfg = ModelConfig::from_file(&files.params).unwrap();

        let pos_chats: Vec<Vec<Message>> =
            serde_json::from_str(&fs::read_to_string(&files.ultrachat_dpo_pos).unwrap()).unwrap();
        let neg_chats: Vec<Vec<Message>> =
            serde_json::from_str(&fs::read_to_string(&files.ultrachat_dpo_neg).unwrap()).unwrap();
        let tmp = TempDir::new("submit_eval_dpo");
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

        let (xp, yp, maskp) = loader_pos.iter().nth(12).unwrap().unwrap();
        let (xn, yn, maskn) = loader_neg.iter().nth(12).unwrap().unwrap();
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
        .unwrap() as f64;
        assert!(heldout_loss < 2.0_f64.ln() - 1e-5);
        assert!(heldout_loss < 0.55);
        assert!(heldout_loss < 0.53);
    }
}
