//! CLI runner for Homework 6: Base Generation, Pretokenization, SFT Chat Tuning, DPO, and Chat.
//!
//! Replaces the interactive download, pretokenization, SFT training, DPO training, and
//! generation cells from `homework/hw6/hw6.py`.
//!
//! Usage (run from `rust/`):
//! ```bash
//! # 1. Generate text from the FineWeb-EDU base pretrained model (`model_base.pth`)
//! cargo run --release -p hw6 --example train_chat_dpo -- --mode base
//!
//! # 2. Pretokenize UltraChat SFT and DPO datasets into JSON token files
//! cargo run --release -p hw6 --example train_chat_dpo -- --mode pretokenize --limit 256
//!
//! # 3. Supervised chat finetuning (SFT) starting from `model_base.pth`
//! cargo run --release -p hw6 --example train_chat_dpo -- --mode sft --max-iter 10 --save /tmp/hw6_sft.safetensors
//!
//! # 4. Direct Preference Optimization (DPO) starting from `model_chat.pth`
//! cargo run --release -p hw6 --example train_chat_dpo -- --mode dpo --max-iter 10 --beta 0.1 --save /tmp/hw6_dpo.safetensors
//!
//! # 5. Interactive chat with `model_chat.pth` (or a saved `.safetensors` checkpoint)
//! cargo run --release -p hw6 --example train_chat_dpo -- --mode chat --prompt "What is the capital of the United States?"
//! ```
//!
//! Set `HW6_DIR=/path/to/dir` if you already have `model_base.pth`, `model_chat.pth`,
//! `params.json`, and the three `ultrachat_*.json` files locally.

use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use candle_core::{Device, Error, Result};
use hw6::{
    ASSISTANT_CLOSE_ID, ASSISTANT_OPEN, Adam, ChatTokenizer, DataLoaderChat, Gpt2Tokenizer,
    Message, ModelConfig, Tokenizer, convert_to_chat_format, download_hw6_files, generate,
    load_pretrained_llm, pretokenize_chat, train_dpo, train_llm_chat,
};

struct Config {
    mode: String,
    prompt: Option<String>,
    checkpoint: Option<PathBuf>,
    save: Option<PathBuf>,
    cache_dir: PathBuf,
    batch_size: usize,
    max_iter: Option<usize>,
    limit: Option<usize>,
    lr: Option<f64>,
    beta: f64,
    temp: f64,
    max_tokens: usize,
}

fn parse_args() -> Config {
    let mut cfg = Config {
        mode: "base".to_string(),
        prompt: None,
        checkpoint: None,
        save: None,
        cache_dir: std::env::temp_dir().join("modernaicourse_hw6"),
        batch_size: 2,
        max_iter: Some(10),
        limit: None,
        lr: None,
        beta: 0.1,
        temp: 0.4,
        max_tokens: 150,
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
            "--max-iter" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse::<usize>() {
                    cfg.max_iter = if v == 0 { None } else { Some(v) };
                }
                i += 2;
            }
            "--limit" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse::<usize>() {
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
            "--beta" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse() {
                    cfg.beta = v;
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
            _ => {
                i += 1;
            }
        }
    }
    cfg
}

fn ensure_pretokenized(
    tokenizer: &ChatTokenizer,
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
        let raw: Vec<Vec<Message>> = serde_json::from_str(
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
            "Pretokenizing first {slice_len} conversations from {} -> {}...",
            src_json.display(),
            dst_json.display()
        );
        pretokenize_chat(tokenizer, &tmp_in, dst_json)?;
        let _ = fs::remove_file(&tmp_in);
    } else {
        eprintln!(
            "Pretokenizing {} -> {}...",
            src_json.display(),
            dst_json.display()
        );
        pretokenize_chat(tokenizer, src_json, dst_json)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let cfg = parse_args();
    let device = Device::Cpu;

    eprintln!("Locating/downloading Homework 6 files from zkolter/Chat-Tuning-Homework...");
    let files = download_hw6_files()?;
    let model_cfg = ModelConfig::from_file(&files.params)?;

    match cfg.mode.as_str() {
        "base" => {
            let ckpt = cfg.checkpoint.as_deref().unwrap_or(&files.model_base);
            eprintln!("Loading base model from {}...", ckpt.display());
            let mut model = load_pretrained_llm(&files.params, ckpt, &device)?;
            let base_tok = Gpt2Tokenizer::new()?;
            let prompt = cfg
                .prompt
                .as_deref()
                .unwrap_or("In a surprising discovery,");
            let prompt_tokens = base_tok.encode(prompt);
            print!("{prompt}");
            io::stdout().flush().map_err(Error::wrap)?;
            generate(
                &mut model,
                &prompt_tokens,
                &base_tok,
                None,
                cfg.temp,
                cfg.max_tokens,
                true,
            )?;
            println!();
        }
        "pretokenize" => {
            let tok = ChatTokenizer::new()?;
            ensure_pretokenized(
                &tok,
                &files.ultrachat_short,
                &cfg.cache_dir.join("ultrachat_tokenized.json"),
                cfg.limit,
            )?;
            ensure_pretokenized(
                &tok,
                &files.ultrachat_dpo_pos,
                &cfg.cache_dir.join("ultrachat_pos_tokenized.json"),
                cfg.limit,
            )?;
            ensure_pretokenized(
                &tok,
                &files.ultrachat_dpo_neg,
                &cfg.cache_dir.join("ultrachat_neg_tokenized.json"),
                cfg.limit,
            )?;
            eprintln!("Pretokenized files written to {}", cfg.cache_dir.display());
        }
        "sft" => {
            let tok = ChatTokenizer::new()?;
            let tok_path = cfg.cache_dir.join("ultrachat_tokenized.json");
            let limit = cfg.limit.or_else(|| {
                cfg.max_iter
                    .map(|iters| (iters + 4) * cfg.batch_size)
            });
            ensure_pretokenized(&tok, &files.ultrachat_short, &tok_path, limit)?;

            let ckpt = cfg.checkpoint.as_deref().unwrap_or(&files.model_base);
            eprintln!("Loading model from {} for SFT...", ckpt.display());
            let mut model = load_pretrained_llm(&files.params, ckpt, &device)?;
            let loader = DataLoaderChat::new(
                &tok_path,
                model_cfg.max_seq_len,
                cfg.batch_size,
                &tok,
                device.clone(),
            )?;

            let lr = cfg.lr.unwrap_or(2e-5);
            let mut opt = Adam::new(model.parameters(), lr, (0.9, 0.95), 1e-8)?;
            eprintln!(
                "Running SFT (batches={}, batch_size={}, lr={lr}, max_iter={:?})...",
                loader.len(),
                cfg.batch_size,
                cfg.max_iter
            );
            train_llm_chat(&mut model, &loader, &mut opt, cfg.max_iter)?;

            if let Some(save_path) = &cfg.save {
                eprintln!("Saving SFT weights to {}...", save_path.display());
                model.save_weights(save_path)?;
            }

            let prompt_text = cfg
                .prompt
                .as_deref()
                .unwrap_or("What is the capital of the United States?");
            let formatted = format!(
                "{}{ASSISTANT_OPEN}",
                convert_to_chat_format(&[Message::user(prompt_text)])
            );
            println!("\nPrompt: {prompt_text}\nAssistant: ");
            let prompt_tokens = tok.encode(&formatted);
            generate(
                &mut model,
                &prompt_tokens,
                &tok,
                Some(ASSISTANT_CLOSE_ID),
                cfg.temp,
                cfg.max_tokens,
                true,
            )?;
            println!();
        }
        "dpo" => {
            let tok = ChatTokenizer::new()?;
            let pos_path = cfg.cache_dir.join("ultrachat_pos_tokenized.json");
            let neg_path = cfg.cache_dir.join("ultrachat_neg_tokenized.json");
            let limit = cfg.limit.or_else(|| {
                cfg.max_iter
                    .map(|iters| (iters + 4) * cfg.batch_size)
            });
            ensure_pretokenized(&tok, &files.ultrachat_dpo_pos, &pos_path, limit)?;
            ensure_pretokenized(&tok, &files.ultrachat_dpo_neg, &neg_path, limit)?;

            let ckpt = cfg.checkpoint.as_deref().unwrap_or(&files.model_chat);
            eprintln!("Loading chat model from {} for DPO...", ckpt.display());
            let mut model = load_pretrained_llm(&files.params, ckpt, &device)?;
            let mut model_ref = model.deep_clone()?;

            let loader_pos = DataLoaderChat::new(
                &pos_path,
                model_cfg.max_seq_len,
                cfg.batch_size,
                &tok,
                device.clone(),
            )?;
            let loader_neg = DataLoaderChat::new(
                &neg_path,
                model_cfg.max_seq_len,
                cfg.batch_size,
                &tok,
                device.clone(),
            )?;

            let lr = cfg.lr.unwrap_or(1e-6);
            let mut opt = Adam::new(model.parameters(), lr, (0.9, 0.95), 1e-8)?;
            eprintln!(
                "Running DPO (pairs={}, batch_size={}, lr={lr}, beta={}, max_iter={:?})...",
                loader_pos.len().min(loader_neg.len()),
                cfg.batch_size,
                cfg.beta,
                cfg.max_iter
            );
            train_dpo(
                &mut model,
                &mut model_ref,
                &loader_pos,
                &loader_neg,
                &mut opt,
                cfg.beta,
                cfg.max_iter,
            )?;

            if let Some(save_path) = &cfg.save {
                eprintln!("Saving DPO weights to {}...", save_path.display());
                model.save_weights(save_path)?;
            }

            let prompt_text = cfg
                .prompt
                .as_deref()
                .unwrap_or("Write a poem about flowers.");
            let formatted = format!(
                "{}{ASSISTANT_OPEN}",
                convert_to_chat_format(&[Message::user(prompt_text)])
            );
            println!("\nPrompt: {prompt_text}\nAssistant: ");
            let prompt_tokens = tok.encode(&formatted);
            generate(
                &mut model,
                &prompt_tokens,
                &tok,
                Some(ASSISTANT_CLOSE_ID),
                cfg.temp,
                cfg.max_tokens,
                true,
            )?;
            println!();
        }
        "chat" => {
            let tok = ChatTokenizer::new()?;
            let ckpt = cfg.checkpoint.as_deref().unwrap_or(&files.model_chat);
            eprintln!("Loading chat model from {}...", ckpt.display());
            let mut model = load_pretrained_llm(&files.params, ckpt, &device)?;

            if let Some(prompt_text) = &cfg.prompt {
                let formatted = format!(
                    "{}{ASSISTANT_OPEN}",
                    convert_to_chat_format(&[Message::user(prompt_text)])
                );
                let prompt_tokens = tok.encode(&formatted);
                generate(
                    &mut model,
                    &prompt_tokens,
                    &tok,
                    Some(ASSISTANT_CLOSE_ID),
                    cfg.temp,
                    cfg.max_tokens,
                    true,
                )?;
                println!();
            } else {
                eprintln!("Interactive chat ready. Type a message (empty line or Ctrl-D to quit).");
                let mut history: Vec<Message> = Vec::new();
                let stdin = io::stdin();
                loop {
                    print!("\nUser> ");
                    io::stdout().flush().map_err(Error::wrap)?;
                    let mut line = String::new();
                    let n = stdin.lock().read_line(&mut line).map_err(Error::wrap)?;
                    let input = line.trim();
                    if n == 0 || input.is_empty() {
                        break;
                    }
                    history.push(Message::user(input));
                    let formatted = format!("{}{ASSISTANT_OPEN}", convert_to_chat_format(&history));
                    let prompt_tokens = tok.encode(&formatted);
                    print!("Assistant> ");
                    io::stdout().flush().map_err(Error::wrap)?;
                    let generated = generate(
                        &mut model,
                        &prompt_tokens,
                        &tok,
                        Some(ASSISTANT_CLOSE_ID),
                        cfg.temp,
                        cfg.max_tokens,
                        true,
                    )?;
                    println!();
                    let reply_ids: Vec<u32> = generated
                        .into_iter()
                        .take_while(|&t| t != ASSISTANT_CLOSE_ID)
                        .collect();
                    let reply = tok.decode(&reply_ids)?;
                    history.push(Message::assistant(reply));
                }
            }
        }
        other => {
            candle_core::bail!(
                "unknown --mode '{other}' (expected: base | pretokenize | sft | dpo | chat)"
            );
        }
    }

    Ok(())
}
