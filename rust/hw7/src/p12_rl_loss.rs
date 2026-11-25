//! Problem 12: Centered policy-gradient RL loss (`rl_loss`).

#[allow(unused_imports)]
use candle_core::{DType, Device, Result, Tensor};

#[allow(unused_imports)]
use crate::{CausalLM, Tokenizer, get_loss_mask, log_probs};

/// Compute the mean-centered policy-gradient (REINFORCE / RLOO-style) scalar loss over
/// `N` sampled completions for one prompt:
///
/// $$\mathcal{L}_{\text{RL}} = -\frac{1}{N_{\text{tok}}} \sum_{i=1}^{N} \log p(y_i \mid x) \cdot \left(R(x, y_i) - \bar{R}\right)$$
///
/// where:
/// - `tokens` has shape `[N, S]` (`DType::U32`): prompt + generated completion tokens
///   (possibly zero-padded after `</ANSWER>`).
/// - `mask` is built row-by-row via [`get_loss_mask`], and shifted to align with next-token
///   targets `y = tokens[:, 1:]` from inputs `x = tokens[:, :-1]`.
/// - $\log p(y_i \mid x)$ is computed by [`log_probs`] over `(logits, tokens[:, 1:], mask[:, 1:])`.
/// - $N_{\text{tok}}$ is the total count of `true` entries in the mask across the batch.
/// - $\bar{R} = \frac{1}{N} \sum_{j=1}^{N} R(x, y_j)$ is the mean reward across completions.
///
/// Note: The leading minus sign is essential so minimizing $\mathcal{L}_{\text{RL}}$ maximizes
/// expected reward (and matches `hw7_tests.py`).
#[allow(unused_variables)]
pub fn rl_loss<M, T>(
    model: &mut M,
    tokenizer: &T,
    tokens: &Tensor,
    rewards: &Tensor,
) -> Result<Tensor>
where
    M: CausalLM,
    T: Tokenizer,
{
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
