//! CLI runner for Homework 7: Base Parallel Sampling, GSM8K Pretokenization, SFT,
//! GSM8K Evaluation, RL Training, and Interactive Problem Solving.
//!
//! Replaces the interactive download, pretokenization, SFT training, evaluation, and
//! RL training cells from `homework/hw7/hw7.py`.
//!
//! Usage (run from `rust/`):
//! ```bash
//! # 1. Parallel sampling from the FineWeb-EDU base pretrained model (`model_base.pth`)
//! cargo run --release -p hw7 --example train_reasoning -- --mode base --num-completions 4
//!
//! # 2. Pretokenize GSM8K train & test splits into JSON token files
//! cargo run --release -p hw7 --example train_reasoning -- --mode pretokenize --limit 256
//!
//! # 3. Supervised finetuning (SFT) on GSM8K starting from `model_base.pth`
//! cargo run --release -p hw7 --example train_reasoning -- --mode sft --epochs 1 --max-iter 10 --save /tmp/hw7_sft.safetensors
//!
//! # 4. Evaluate `model_sft.pth` (or a saved `.safetensors` checkpoint) on GSM8K test problems
//! cargo run --release -p hw7 --example train_reasoning -- --mode eval --num-completions 8 --max-cases 10
//!
//! # 5. Reinforcement learning (RL) starting from `model_sft.pth`
//! cargo run --release -p hw7 --example train_reasoning -- --mode rl --num-completions 8 --max-iter 10 --save /tmp/hw7_rl.safetensors
//!
//! # 6. Solve a single math word problem with live `<TOOL>...</TOOL>` evaluation
//! cargo run --release -p hw7 --example train_reasoning -- --mode solve \
//!   --prompt "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
//! ```
//!
//! Set `HW7_DIR=/path/to/dir` if you already have `model_base.pth`, `model_sft.pth`,
//! `params.json`, `gsm8k_train.json`, and `gsm8k_test.json` locally.

use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{Device, Error, Result};
use hw7::{
    ANSWER_CLOSE_ID, Adam, DataLoader, Gpt2Tokenizer, Gsm8kExample, QUESTION_CLOSE, QUESTION_OPEN,
    ReasoningTokenizer, THINK_OPEN, Tokenizer, download_hw7_files, eval_model, extract_answer,
    generate_parallel, generate_parallel_tool, load_pretrained_llm, pretokenize_gsm8k,
    train_llm_rl, train_llm_sft,
};

struct Config {
    mode: String,
    prompt: Option<String>,
    checkpoint: Option<PathBuf>,
    save: Option<PathBuf>,
    cache_dir: PathBuf,
    batch_size: usize,
    epochs: usize,
    max_iter: Option<usize>,
    max_cases: usize,
    limit: Option<usize>,
    lr: Option<f64>,
    num_completions: usize,
    temp: f64,
    max_tokens: usize,
    correct_weight: f64,
    format_weight: f64,
}

fn parse_args() -> Config {
    let mut cfg = Config {
        mode: "base".to_string(),
        prompt: None,
        checkpoint: None,
        save: None,
        cache_dir: std::env::temp_dir().join("modernaicourse_hw7"),
        batch_size: 4,
        epochs: 1,
        max_iter: Some(10),
        max_cases: 10,
        limit: None,
        lr: None,
        num_completions: 4,
        temp: 0.7,
        max_tokens: 200,
        correct_weight: 1.0,
        format_weight: 0.05,
    };

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" if i + 1 < args.len() => {
                cfg.mode = args[i + 1].clone();
                i += 2;
            }
            "--prompt" if i + 1 < args.len() => {
                cfg.prompt = Some(args[i + 1].clone());
                i += 2;
            }
            "--checkpoint" if i + 1 < args.len() => {
                cfg.checkpoint = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--save" if i + 1 < args.len() => {
                cfg.save = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--cache-dir" if i + 1 < args.len() => {
                cfg.cache_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--batch-size" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse() {
                    cfg.batch_size = v;
                }
                i += 2;
            }
            "--epochs" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse() {
                    cfg.epochs = v;
                }
                i += 2;
            }
            "--max-iter" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse::<usize>() {
                    cfg.max_iter = if v == 0 { None } else { Some(v) };
                }
                i += 2;
            }
            "--max-cases" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse() {
                    cfg.max_cases = v;
                }
                i += 2;
            }
            "--limit" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse() {
                    cfg.limit = Some(v);
                }
                i += 2;
            }
            "--lr" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse() {
                    cfg.lr = Some(v);
                }
                i += 2;
            }
            "--num-completions" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse() {
                    cfg.num_completions = v;
                }
                i += 2;
            }
            "--temp" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse() {
                    cfg.temp = v;
                }
                i += 2;
            }
            "--max-tokens" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse() {
                    cfg.max_tokens = v;
                }
                i += 2;
            }
            "--correct-weight" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse() {
                    cfg.correct_weight = v;
                }
                i += 2;
            }
            "--format-weight" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse() {
                    cfg.format_weight = v;
                }
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }
    cfg
}

fn ensure_pretokenized(
    tokenizer: &ReasoningTokenizer,
    src_json: &Path,
    dst_json: &Path,
    limit: Option<usize>,
) -> Result<()> {
    if dst_json.is_file() && limit.is_none() {
        return Ok(());
    }
    if let Some(parent) = dst_json.parent() {
        fs::create_dir_all(parent).map_err(Error::wrap)?;
    }

    if let Some(n) = limit {
        let raw: Vec<Gsm8kExample> = serde_json::from_str(
            &fs::read_to_string(src_json).map_err(Error::wrap)?,
        )
        .map_err(|e| Error::Msg(format!("failed to parse {}: {e}", src_json.display())))?;
        let slice_len = n.min(raw.len());
        let tmp_in = dst_json.with_extension("subset.json");
        fs::write(
            &tmp_in,
            serde_json::to_string(&raw[..slice_len])
                .map_err(|e| Error::Msg(format!("failed to serialize subset: {e}")))?,
        )
        .map_err(Error::wrap)?;
        eprintln!(
            "Pretokenizing first {slice_len} GSM8K examples from {} -> {}...",
            src_json.display(),
            dst_json.display()
        );
        pretokenize_gsm8k(tokenizer, &tmp_in, dst_json)?;
        let _ = fs::remove_file(&tmp_in);
    } else {
        eprintln!(
            "Pretokenizing {} -> {}...",
            src_json.display(),
            dst_json.display()
        );
        pretokenize_gsm8k(tokenizer, src_json, dst_json)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let cfg = parse_args();
    let device = Device::Cpu;

    eprintln!("Locating/downloading Homework 7 checkpoints & GSM8K dataset...");
    let files = download_hw7_files()?;

    match cfg.mode.as_str() {
        "base" => {
            let ckpt = cfg.checkpoint.as_deref().unwrap_or(&files.model_base);
            eprintln!("Loading base model from {}...", ckpt.display());
            let mut model = load_pretrained_llm(&files.params, ckpt, &device)?;
            let base_tok = Gpt2Tokenizer::new()?;
            let prompt = cfg.prompt.as_deref().unwrap_or("In a shocking discovery,");
            let prompt_tokens = base_tok.encode(prompt);
            let max_toks = cfg.max_tokens.min(100);
            let generated = generate_parallel(
                &mut model,
                &prompt_tokens,
                cfg.num_completions,
                None,
                cfg.temp,
                max_toks,
            )?;
            for (idx, row) in generated.to_vec2::<u32>()?.into_iter().enumerate() {
                let text = base_tok.decode(&row)?;
                println!("=== Completion {} ===\n{}\n", idx + 1, text);
            }
        }
        "pretokenize" => {
            let tok = ReasoningTokenizer::new()?;
            ensure_pretokenized(
                &tok,
                &files.gsm8k_train,
                &cfg.cache_dir.join("gsm8k_train_tokenized.json"),
                cfg.limit,
            )?;
            ensure_pretokenized(
                &tok,
                &files.gsm8k_test,
                &cfg.cache_dir.join("gsm8k_test_tokenized.json"),
                cfg.limit,
            )?;
            eprintln!("Pretokenized GSM8K files written to {}", cfg.cache_dir.display());
        }
        "sft" => {
            let tok = ReasoningTokenizer::new()?;
            let train_tok = cfg.cache_dir.join("gsm8k_train_tokenized.json");
            let limit = cfg.limit.or_else(|| {
                cfg.max_iter
                    .map(|iters| (iters + 4) * cfg.batch_size)
            });
            ensure_pretokenized(&tok, &files.gsm8k_train, &train_tok, limit)?;

            let ckpt = cfg.checkpoint.as_deref().unwrap_or(&files.model_base);
            eprintln!("Loading model from {} for GSM8K SFT...", ckpt.display());
            let mut model = load_pretrained_llm(&files.params, ckpt, &device)?;
            let loader = DataLoader::new(&train_tok, cfg.batch_size, &tok, device.clone())?;

            let lr = cfg.lr.unwrap_or(3e-5);
            let mut opt = Adam::new(model.parameters(), lr, (0.9, 0.95), 1e-8)?;
            for epoch in 0..cfg.epochs {
                eprintln!(
                    "SFT epoch {}/{} (batches={}, batch_size={}, lr={lr}, max_iter={:?})...",
                    epoch + 1,
                    cfg.epochs,
                    loader.len(),
                    cfg.batch_size,
                    cfg.max_iter
                );
                train_llm_sft(&mut model, &loader, &mut opt, cfg.max_iter)?;
            }

            if let Some(save_path) = &cfg.save {
                eprintln!("Saving SFT weights to {}...", save_path.display());
                model.save_weights(save_path)?;
            }
        }
        "eval" => {
            let tok = ReasoningTokenizer::new()?;
            let test_tok = cfg.cache_dir.join("gsm8k_test_tokenized.json");
            let limit = cfg.limit.or(Some(cfg.max_cases + 4));
            ensure_pretokenized(&tok, &files.gsm8k_test, &test_tok, limit)?;

            let ckpt = cfg.checkpoint.as_deref().unwrap_or(&files.model_sft);
            eprintln!("Evaluating model from {} on GSM8K...", ckpt.display());
            let mut model = load_pretrained_llm(&files.params, ckpt, &device)?;
            let loader = DataLoader::new(&test_tok, 1, &tok, device)?;

            let (acc, fmt, passk) = eval_model(
                &loader,
                &mut model,
                &tok,
                cfg.num_completions,
                cfg.max_tokens,
                cfg.temp,
                cfg.max_cases,
            )?;
            println!(
                "Accuracy (Pass@1): {:.4}\nPass@{}: {:.4}\nCorrect format: {:.4}",
                acc, cfg.num_completions, passk, fmt
            );
        }
        "rl" => {
            let tok = ReasoningTokenizer::new()?;
            let train_tok = cfg.cache_dir.join("gsm8k_train_tokenized.json");
            let limit = cfg.limit.or_else(|| cfg.max_iter.map(|n| n + 4));
            ensure_pretokenized(&tok, &files.gsm8k_train, &train_tok, limit)?;

            let ckpt = cfg.checkpoint.as_deref().unwrap_or(&files.model_sft);
            eprintln!("Loading model from {} for RL training...", ckpt.display());
            let mut model = load_pretrained_llm(&files.params, ckpt, &device)?;
            let loader = DataLoader::new(&train_tok, 1, &tok, device)?;

            let lr = cfg.lr.unwrap_or(2e-6);
            let mut opt = Adam::new(model.parameters(), lr, (0.9, 0.95), 1e-8)?;
            eprintln!(
                "Running RL (num_completions={}, max_tokens={}, lr={lr}, max_iter={:?})...",
                cfg.num_completions, cfg.max_tokens, cfg.max_iter
            );
            train_llm_rl(
                &mut model,
                &loader,
                &mut opt,
                &tok,
                cfg.num_completions,
                cfg.temp,
                cfg.max_tokens,
                cfg.max_iter,
                cfg.correct_weight,
                cfg.format_weight,
            )?;

            if let Some(save_path) = &cfg.save {
                eprintln!("Saving RL weights to {}...", save_path.display());
                model.save_weights(save_path)?;
            }
        }
        "solve" => {
            let tok = ReasoningTokenizer::new()?;
            let ckpt = cfg.checkpoint.as_deref().unwrap_or(&files.model_sft);
            eprintln!("Loading reasoning model from {}...", ckpt.display());
            let mut model = load_pretrained_llm(&files.params, ckpt, &device)?;

            let question = cfg.prompt.as_deref().unwrap_or(
                "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?",
            );
            let prompt_str = format!("{QUESTION_OPEN}{question}{QUESTION_CLOSE}{THINK_OPEN}");
            let prompt_tokens = tok.encode(&prompt_str);

            let tokens = generate_parallel_tool(
                &mut model,
                &prompt_tokens,
                &tok,
                cfg.num_completions,
                Some(ANSWER_CLOSE_ID),
                cfg.temp,
                cfg.max_tokens,
            )?;
            for (idx, row) in tokens.to_vec2::<u32>()?.into_iter().enumerate() {
                let ans = extract_answer(&tok, &row);
                let trimmed: Vec<u32> = row.into_iter().take_while(|&t| t != 0).collect();
                let decoded = tok.decode(&trimmed)?;
                println!(
                    "=== Sample {} (extracted answer: {:?}) ===\n{}\n",
                    idx + 1,
                    ans,
                    decoded
                );
            }
        }
        other => {
            candle_core::bail!(
                "unknown --mode '{other}' (expected: base | pretokenize | sft | eval | rl | solve)"
            );
        }
    }

    Ok(())
}
