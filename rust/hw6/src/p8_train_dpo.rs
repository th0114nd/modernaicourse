//! Problem 8: Direct Preference Optimization (DPO) training loop.

#[allow(unused_imports)]
use candle_core::{Result, Tensor};

#[allow(unused_imports)]
use crate::{CausalLM, Optimizer, dpo_loss};

/// Run one pass of DPO finetuning over paired preferred (`loader_pos`) and dispreferred
/// (`loader_neg`) chat minibatches.
///
/// # Algorithm
/// For each `(step, (pos_batch, neg_batch))` in `loader_pos.into_iter().zip(loader_neg)`:
/// 1. If `max_iter` is `Some(limit)` and `step >= limit`, stop immediately (`break`).
/// 2. Compute per-example DPO losses via
///    `dpo_loss(model, model_ref, &xp, &yp, &maskp, &xn, &yn, &maskn, beta)?`
///    and average them into a scalar via `.mean_all()?`.
/// 3. Call `opt.zero_grad();`, `let grads = loss.backward()?;`, and `opt.step(&grads)?;`.
#[allow(unused_variables)]
pub fn train_dpo<M, R, IP, IN, O>(
    model: &mut M,
    model_ref: &mut R,
    loader_pos: IP,
    loader_neg: IN,
    opt: &mut O,
    beta: f64,
    max_iter: Option<usize>,
) -> Result<()>
where
    M: CausalLM,
    R: CausalLM,
    IP: IntoIterator<Item = Result<(Tensor, Tensor, Tensor)>>,
    IN: IntoIterator<Item = Result<(Tensor, Tensor, Tensor)>>,
    O: Optimizer,
{
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
