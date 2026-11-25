//! Problem 8: Arithmetic tool evaluator (`eval_tool`).

#[allow(unused_imports)]
use crate::eval_arithmetic_expr;

/// Evaluate an arithmetic expression generated inside `<TOOL>...</TOOL>` tags and format
/// the response string for `<RESPONSE>...</RESPONSE>`.
///
/// # Requirements (matching `homework/hw7/hw7.py`)
/// 1. Evaluate `tool_call_text` using the provided [`eval_arithmetic_expr`] helper
///    (Rust counterpart to Python's `eval()` for numeric expressions).
/// 2. If evaluation fails (invalid syntax, division by zero, non-finite result, etc.),
///    return `"ERROR".to_string()`.
/// 3. If the numeric value `v` is within `1e-4` of an integer (`(v - v.round()).abs() <= 1e-4`,
///    and within `i64` range), return the rounded integer string (e.g., `"2"`, `"3"`, `"24"`).
/// 4. Otherwise, return the decimal string (e.g., `"3.5"`, `"1.75"`).
#[allow(unused_variables)]
pub fn eval_tool(tool_call_text: &str) -> String {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
