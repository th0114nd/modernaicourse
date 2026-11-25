//! Problem 7b: Direct Preference Optimization (DPO) loss.

#[allow(unused_imports)]
use candle_core::{DType, Result, Tensor};

use crate::CausalLM;
#[allow(unused_imports)]
use crate::{log_probs, softplus};

/// Compute the per-example Direct Preference Optimization (DPO) loss:
///
/// $$L_{\text{DPO}} = \text{softplus}\!\left(-\log\frac{p(y^+ \mid x)}{p_{\text{ref}}(y^+ \mid x)} + \log\frac{p(y^- \mid x)}{p_{\text{ref}}(y^- \mid x)},\; \beta\right)$$
///
/// # Inputs
/// - `model`: policy model being optimized (receives gradients)
/// - `model_ref`: fixed reference model (must **not** receive gradients)
/// - `xp`, `yp`, `maskp`: preferred completion inputs, targets, and loss mask (`[B, S]`)
/// - `xn`, `yn`, `maskn`: dispreferred completion inputs, targets, and loss mask (`[B, S]`)
/// - `beta`: inverse-temperature DPO hyperparameter $\beta$ (typically `0.1`)
///
/// # Output
/// 1D tensor of shape `[batch_size]` containing the DPO loss for each preference pair.
///
/// # Candle No-Grad Note
/// Candle does not have a `torch.no_grad()` context manager. Call `.detach()` on the
/// reference model's forward outputs (and/or reference log probabilities) so that
/// `loss.backward()` computes gradients **only** for `model` and leaves `grads.get(ref_param)`
/// as `None`.
#[allow(unused_variables)]
#[allow(clippy::too_many_arguments)]
pub fn dpo_loss<M: CausalLM, R: CausalLM>(
    model: &mut M,
    model_ref: &mut R,
    xp: &Tensor,
    yp: &Tensor,
    maskp: &Tensor,
    xn: &Tensor,
    yn: &Tensor,
    maskn: &Tensor,
    beta: f64,
) -> Result<Tensor> {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
