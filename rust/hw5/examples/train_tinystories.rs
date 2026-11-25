//! CLI runner for Homework 5: BPE Tokenizer Demo & TinyStories LLM Pretraining.
//!
//! Replaces the interactive TinyStories BPE, pretokenization, training (Q13 & Q14),
//! and generation cells from `homework/hw5/hw5.py`.
//!
//! Usage (run from `rust/`):
//! ```bash
//! # 1. Quick CPU smoke run (tiny model on 64K chars of TinyStories)
//! cargo run --release -p hw5 --example train_tinystories -- --mode tiny
//!
//! # 2. BPE tokenizer demo on TinyStories (100K chars -> 2000 tokens, round-trip check)
//! cargo run --release -p hw5 --example train_tinystories -- --mode bpe
//!
//! # 3. Q13 small LLM (2 chunks of 1M chars, 4 layers, dim=256, seq_len=512, batch=16)
//! cargo run --release -p hw5 --example train_tinystories -- --mode small
//!
//! # 4. Q14 121M-parameter LLM on full TinyStories (recommended with GPU)
//! cargo run --release -p hw5 --example train_tinystories -- --mode full
//!
//! # Custom prompt / sampling flags
//! cargo run --release -p hw5 --example train_tinystories -- --mode tiny --prompt "Once upon a time," --temp 0.7 --max-tokens 100
//! ```
//!
//! Set `TINYSTORIES_PATH=/path/to/TinyStoriesV2-GPT4-train.txt` if you already have the
//! dataset locally to skip downloading from HuggingFace Hub.

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use candle_core::{Device, Error, Result};
use hw5::{
    Adam, DataLoader, Gpt2Tokenizer, LLM, Tokenizer, bpe_decode, bpe_encode, download_tinystories,
    generate, padded_vocab_size, pretokenize_data, read_char_chunk, train_bpe, train_llm,
};

struct Config {
    mode: String,
    prompt: String,
    temp: f64,
    max_tokens: usize,
}

fn parse_args() -> Config {
    let mut cfg = Config {
        mode: "small".to_string(),
        prompt: "Once upon a time,".to_string(),
        temp: 0.7,
        max_tokens: 200,
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
                cfg.prompt = args[i + 1].clone();
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

fn run_bpe_demo(txt_path: &PathBuf) -> Result<()> {
    eprintln!("Reading 100,000 characters from {} for BPE training...", txt_path.display());
    let file = File::open(txt_path).map_err(Error::wrap)?;
    let mut reader = BufReader::new(file);
    let train_text = read_char_chunk(&mut reader, 100_000).map_err(Error::wrap)?;
    let test_text = read_char_chunk(&mut reader, 1_000).map_err(Error::wrap)?;

    let (tokens, merges) = train_bpe(&train_text, 2_000);
    let encoded = bpe_encode(&test_text, &merges, &tokens);
    let decoded = bpe_decode(&encoded, &tokens);

    println!("Original text ({} chars):\n{test_text}", test_text.len());
    println!("\nEncoded into {} tokens: {:?}", encoded.len(), &encoded[..encoded.len().min(32)]);
    assert_eq!(test_text, decoded, "BPE round-trip decode mismatch!");
    println!("\nBPE round-trip verified successfully!");
    Ok(())
}

fn main() -> Result<()> {
    let cfg = parse_args();
    eprintln!("Locating/downloading TinyStoriesV2-GPT4-train.txt...");
    let txt_path = download_tinystories()?;

    if cfg.mode == "bpe" {
        return run_bpe_demo(&txt_path);
    }

    let device = Device::Cpu;
    let tokenizer = Gpt2Tokenizer::new()?;
    let vocab_size = padded_vocab_size(tokenizer.n_vocab());

    let (bin_name, chunk_size, max_chunks, seq_len, batch_size, dim, n_heads, ffn_dim, num_layers) =
        match cfg.mode.as_str() {
            "tiny" => (
                "TinyStoriesV2-GPT4-train.tiny.bin",
                65_536usize,
                Some(1usize),
                64usize,
                8usize,
                64usize,
                4usize,
                128usize,
                2usize,
            ),
            "small" => (
                "TinyStoriesV2-GPT4-train.small.bin",
                1usize << 20,
                Some(2usize),
                512usize,
                16usize,
                256usize,
                8usize,
                512usize,
                4usize,
            ),
            "full" => (
                "TinyStoriesV2-GPT4-train.bin",
                1usize << 20,
                None,
                1024usize,
                32usize,
                768usize,
                12usize,
                4 * 768,
                6usize,
            ),
            other => {
                candle_core::bail!("unknown --mode '{other}' (expected: bpe | tiny | small | full)");
            }
        };

    let bin_path = std::env::temp_dir().join(bin_name);
    eprintln!(
        "Pretokenizing {} -> {} (chunk_size={chunk_size}, max_chunks={max_chunks:?})...",
        txt_path.display(),
        bin_path.display()
    );
    pretokenize_data(&tokenizer, &txt_path, &bin_path, chunk_size, max_chunks)?;

    let loader = DataLoader::new(&bin_path, seq_len, batch_size, device.clone())?;
    eprintln!(
        "Training LLM (mode={}, batches={}, seq_len={seq_len}, batch_size={batch_size}, dim={dim}, layers={num_layers})...",
        cfg.mode,
        loader.len()
    );

    let mut model = LLM::new(vocab_size, dim, n_heads, seq_len, ffn_dim, num_layers, &device)?;
    let mut opt = Adam::new(model.parameters(), 1e-3, (0.9, 0.95), 1e-8)?;
    train_llm(&mut model, &loader, &mut opt)?;

    let prompt_tokens = tokenizer.encode(&cfg.prompt);
    println!("\n--- Generation (temp={}, max_tokens={}) ---", cfg.temp, cfg.max_tokens);
    print!("{}", cfg.prompt);
    let generated = generate(
        &mut model,
        &prompt_tokens,
        &tokenizer,
        cfg.temp,
        cfg.max_tokens,
        true,
    )?;
    println!("\n--- Generated {} tokens ---", generated.len());
    Ok(())
}
