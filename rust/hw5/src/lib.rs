//! Homework 5: BPE Tokenizer from Scratch & Training an LLM on TinyStories.
//!
//! See `README.md` for full problem descriptions.

pub mod nn;
pub mod tinystories;

pub mod p1_text_to_corpus;
pub mod p2_most_common_pair;
pub mod p3_merge_pair;
pub mod p4_train_bpe;
pub mod p5a_bpe_encode;
pub mod p5b_bpe_decode;

pub mod p6a_linear;
pub mod p6b_embedding;
pub mod p6c_silu;
pub mod p6d_rms_norm;
pub mod p6e_self_attention;
pub mod p6f_multi_head_attention_kv_cache;
pub mod p6g_mlp;
pub mod p6h_transformer_block;
pub mod p6i_llm;

pub mod p7_cross_entropy_loss;
pub mod p8_pretokenize_data;
pub mod p9_dataloader;
pub mod p10_adam;
pub mod p11_train_llm;
pub mod p12_generate;

pub use nn::{CausalLM, Optimizer, multinomial, read_char_chunk, split_keep_whitespace};
pub use tinystories::{
    Gpt2Tokenizer, TINYSTORIES_REPO_ID, TINYSTORIES_TRAIN_FILE, Tokenizer, download_tinystories,
    padded_vocab_size,
};

pub use p1_text_to_corpus::text_to_corpus;
pub use p2_most_common_pair::most_common_pair;
pub use p3_merge_pair::merge_pair;
pub use p4_train_bpe::train_bpe;
pub use p5a_bpe_encode::bpe_encode;
pub use p5b_bpe_decode::bpe_decode;

pub use p6a_linear::Linear;
pub use p6b_embedding::Embedding;
pub use p6c_silu::silu;
pub use p6d_rms_norm::rms_norm;
pub use p6e_self_attention::self_attention;
pub use p6f_multi_head_attention_kv_cache::MultiHeadAttentionKVCache;
pub use p6g_mlp::MLP;
pub use p6h_transformer_block::TransformerBlock;
pub use p6i_llm::LLM;

pub use p7_cross_entropy_loss::cross_entropy_loss;
pub use p8_pretokenize_data::pretokenize_data;
pub use p9_dataloader::{DataLoader, DataLoaderIter};
pub use p10_adam::Adam;
pub use p11_train_llm::train_llm;
pub use p12_generate::{generate, generate_with_writer};
