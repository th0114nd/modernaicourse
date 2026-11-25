//! Homework 4: Transformers and `Llama3Simplified` in Candle.
//!
//! See `README.md` for full problem descriptions.

pub mod llama;
pub mod tokenizer;

pub mod p1_linear;
pub mod p2_embedding;
pub mod p3_silu;
pub mod p4_rms_norm;
pub mod p5_self_attention;
pub mod p6a_multi_head_attention;
pub mod p6b_multi_head_attention_kv_cache;
pub mod p7_gated_mlp;
pub mod p8_transformer_block;
pub mod p9_llama3_simplified;
pub mod p10_generate;

pub use llama::{
    CausalLM, LLAMA_REPO_ID, LlamaConfig, LlamaFiles, download_llama_files, load_pretrained_llama3,
    load_pth_checkpoint, multinomial,
};
pub use tokenizer::{ChatFormat, LlamaTokenizer, Message, Tokenizer};

pub use p1_linear::Linear;
pub use p2_embedding::Embedding;
pub use p3_silu::silu;
pub use p4_rms_norm::RMSNorm;
pub use p5_self_attention::self_attention;
pub use p6a_multi_head_attention::MultiHeadAttention;
pub use p6b_multi_head_attention_kv_cache::MultiHeadAttentionKVCache;
pub use p7_gated_mlp::GatedMLP;
pub use p8_transformer_block::TransformerBlock;
pub use p9_llama3_simplified::Llama3Simplified;
pub use p10_generate::{generate, generate_with_writer};
