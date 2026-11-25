//! Problem 2: Reverse-mode automatic differentiation (`compute_gradients`).

use crate::autograd::Variable;

/// Recursively compute gradients in a computation graph starting from `var`.
///
/// Algorithm:
/// 1. If `var.grad()` is `None` (i.e. this is the output node being differentiated),
///    set it to `1.0` via `var.set_grad(1.0)`.
/// 2. If `var.function()` is `Some(func)`:
///    - Collect the parent scalar values `[p.value() for p in var.parents()]` and
///      call `func.backward(grad, &parent_values)`.
///    - For each `(parent, g)` pair:
///      - Accumulate `g` into `parent.grad()` (set to `g` if `None`, otherwise add `g`).
///      - Decrement `parent.num_children()` by `1` via `parent.set_num_children(...)`.
///      - If the parent's `num_children` reaches `0`, recursively call `compute_gradients(&parent)`.
#[allow(unused_variables)]
pub fn compute_gradients(var: &Variable) {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
