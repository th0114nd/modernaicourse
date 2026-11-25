//! Homework 7: Reasoning Models & Reinforcement Learning on GSM8K.
//!
//! See `README.md` for full problem descriptions.

pub mod gsm8k;
pub mod nn;

pub mod p0a_linear;
pub mod p0b_embedding;
pub mod p0c_silu;
pub mod p0d_rms_norm;
pub mod p0e_self_attention;
pub mod p0f_mlp;
pub mod p0g_transformer_block;
pub mod p0h_cross_entropy_loss;
pub mod p0i_adam;
pub mod p0j_llm;
pub mod p0k_log_probs;

pub mod p1_multi_head_attention_kv_cache;
pub mod p2_generate_parallel;
pub mod p3_gsm8k_to_text;
pub mod p4_pretokenize_gsm8k;
pub mod p5_get_loss_mask;
pub mod p6_dataloader;
pub mod p7_train_llm_sft;
pub mod p8_eval_tool;
pub mod p9_generate_parallel_tool;
pub mod p10a_extract_answer;
pub mod p10b_grade_responses;
pub mod p11_eval_model;
pub mod p12_rl_loss;
pub mod p13_train_llm_rl;

pub use gsm8k::{
    ANSWER_CLOSE, ANSWER_CLOSE_ID, ANSWER_OPEN, ANSWER_OPEN_ID, EOT_ID, Gpt2Tokenizer,
    Gsm8kExample, HW7_REPO_ID, Hw7Files, ModelConfig, QUESTION_CLOSE, QUESTION_CLOSE_ID,
    QUESTION_OPEN, QUESTION_OPEN_ID, RESPONSE_CLOSE, RESPONSE_CLOSE_ID, RESPONSE_OPEN,
    RESPONSE_OPEN_ID, ReasoningTokenizer, THINK_CLOSE, THINK_CLOSE_ID, THINK_OPEN, THINK_OPEN_ID,
    TOOL_CLOSE, TOOL_CLOSE_ID, TOOL_OPEN, TOOL_OPEN_ID, Tokenizer, download_gsm8k_datasets,
    download_hw7_files, download_hw7_models, eval_arithmetic_expr, load_checkpoint_tensors,
    load_pretrained_llm,
};
pub use nn::{CausalLM, Optimizer, mask_indices, multinomial, multinomial_batch};

pub use p0a_linear::Linear;
pub use p0b_embedding::Embedding;
pub use p0c_silu::silu;
pub use p0d_rms_norm::rms_norm;
pub use p0e_self_attention::self_attention;
pub use p0f_mlp::MLP;
pub use p0g_transformer_block::TransformerBlock;
pub use p0h_cross_entropy_loss::cross_entropy_loss;
pub use p0i_adam::Adam;
pub use p0j_llm::LLM;
pub use p0k_log_probs::log_probs;

pub use p1_multi_head_attention_kv_cache::{
    DEFAULT_MAX_CACHE_BATCHES, MultiHeadAttentionKVCache,
};
pub use p2_generate_parallel::generate_parallel;
pub use p3_gsm8k_to_text::gsm8k_to_text;
pub use p4_pretokenize_gsm8k::pretokenize_gsm8k;
pub use p5_get_loss_mask::get_loss_mask;
pub use p6_dataloader::{DataLoader, DataLoaderIter};
pub use p7_train_llm_sft::train_llm_sft;
pub use p8_eval_tool::eval_tool;
pub use p9_generate_parallel_tool::generate_parallel_tool;
pub use p10a_extract_answer::extract_answer;
pub use p10b_grade_responses::grade_responses;
pub use p11_eval_model::eval_model;
pub use p12_rl_loss::rl_loss;
pub use p13_train_llm_rl::train_llm_rl;
