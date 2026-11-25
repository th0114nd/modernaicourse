//! Interactive / CLI Llama 3.2 1B Simplified inference runner.
//!
//! Replaces the final chat-generation cell in `homework/hw4/hw4.py`.
//!
//! Usage (run from `rust/`):
//! ```bash
//! # Default prompt ("Why is the sky blue?"), temp=0.7, max_tokens=200
//! cargo run --release -p hw4 --example chat
//!
//! # Custom prompt, temperature, and max tokens
//! cargo run --release -p hw4 --example chat -- "Explain attention in two sentences." 0.7 150
//! ```
//!
//! Note: First run downloads ~4.9 GB (`consolidated.00.pth`, `params.json`, `tokenizer.model`)
//! from `zkolter/Llama-3.2-1B-Instruct-Simplified` into the HuggingFace cache.
//! Set `LLAMA_DIR=/path/to/dir` if you already have those three files locally.

use candle_core::{Device, Result};
use hw4::{ChatFormat, LlamaTokenizer, Message, generate, load_pretrained_llama3};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let prompt_text = args
        .first()
        .map(String::as_str)
        .unwrap_or("Why is the sky blue?");
    let temp: f64 = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.7);
    let max_tokens: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);

    let device = Device::Cpu;
    eprintln!("Loading Llama-3.2-1B-Instruct-Simplified (this may download ~4.9 GB on first run)...");
    let (mut model, files) = load_pretrained_llama3(&device)?;
    let tokenizer = LlamaTokenizer::from_file(&files.tokenizer_model)?;
    let chat = ChatFormat::new(&tokenizer);

    let dialog = vec![Message::user(prompt_text)];
    let prompt_tokens = chat.encode_dialog_prompt(&dialog);

    eprintln!(
        "Prompt ({} tokens): {prompt_text}\n--- Response (temp={temp}, max_tokens={max_tokens}) ---",
        prompt_tokens.len()
    );
    let generated = generate(&mut model, &prompt_tokens, &tokenizer, temp, max_tokens, true)?;
    println!();
    eprintln!("--- Generated {} tokens ---", generated.len());
    Ok(())
}
