//! Homework 6: Supervised Finetuning for Chat & Direct Preference Optimization (DPO).
//!
//! See `README.md` for full problem descriptions.

pub mod chat;
pub mod nn;

pub mod p0a_linear;
pub mod p0b_embedding;
pub mod p0c_silu;
pub mod p0d_rms_norm;
pub mod p0e_self_attention;
pub mod p0f_multi_head_attention_kv_cache;
pub mod p0g_mlp;
pub mod p0h_transformer_block;
pub mod p0i_cross_entropy_loss;
pub mod p0j_adam;
pub mod p0k_llm;
pub mod p0l_generate;

pub mod p1_convert_to_chat_format;
pub mod p2_pretokenize_chat;
pub mod p3_get_loss_mask;
pub mod p4_dataloader_chat;
pub mod p5_train_llm_chat;

pub mod p6_log_probs;
pub mod p7a_softplus;
pub mod p7b_dpo_loss;
pub mod p8_train_dpo;

pub use chat::{
    ASSISTANT_CLOSE, ASSISTANT_CLOSE_ID, ASSISTANT_OPEN, ASSISTANT_OPEN_ID, ChatFiles,
    ChatTokenizer, EOT_ID, Gpt2Tokenizer, HW6_REPO_ID, Message, ModelConfig, Tokenizer, USER_CLOSE,
    USER_CLOSE_ID, USER_OPEN, USER_OPEN_ID, download_hw6_files, load_checkpoint_tensors,
    load_pretrained_llm,
};
pub use nn::{CausalLM, Optimizer, mask_indices, multinomial};

pub use p0a_linear::Linear;
pub use p0b_embedding::Embedding;
pub use p0c_silu::silu;
pub use p0d_rms_norm::rms_norm;
pub use p0e_self_attention::self_attention;
pub use p0f_multi_head_attention_kv_cache::MultiHeadAttentionKVCache;
pub use p0g_mlp::MLP;
pub use p0h_transformer_block::TransformerBlock;
pub use p0i_cross_entropy_loss::cross_entropy_loss;
pub use p0j_adam::Adam;
pub use p0k_llm::LLM;
pub use p0l_generate::{generate, generate_with_writer};

pub use p1_convert_to_chat_format::convert_to_chat_format;
pub use p2_pretokenize_chat::pretokenize_chat;
pub use p3_get_loss_mask::get_loss_mask;
pub use p4_dataloader_chat::{DataLoaderChat, DataLoaderChatIter};
pub use p5_train_llm_chat::train_llm_chat;

pub use p6_log_probs::log_probs;
pub use p7a_softplus::softplus;
pub use p7b_dpo_loss::dpo_loss;
pub use p8_train_dpo::train_dpo;
