//! Problem 3: Converting a GSM8K example into Homework 7's tagged reasoning format (`gsm8k_to_text`).

#[allow(unused_imports)]
use regex::Regex;

#[allow(unused_imports)]
use crate::{
    ANSWER_CLOSE, ANSWER_OPEN, Gsm8kExample, QUESTION_CLOSE, QUESTION_OPEN, RESPONSE_CLOSE,
    RESPONSE_OPEN, THINK_CLOSE, THINK_OPEN, TOOL_CLOSE, TOOL_OPEN,
};

/// Convert one [`Gsm8kExample`] (`question`, `answer`) into the tagged reasoning format:
///
/// ```text
/// <QUESTION>{question}</QUESTION><THINK>{reasoning}</THINK><ANSWER>{final_answer}</ANSWER>
/// ```
///
/// Within `{reasoning}`, every GSM8K calculator annotation `<<expr=result>>` is replaced by:
///
/// ```text
/// <TOOL>{expr}</TOOL><RESPONSE>{result}</RESPONSE>
/// ```
///
/// The final integer answer follows `\n#### ` at the end of `example.answer` (stripped of the
/// `#### ` delimiter and surrounding whitespace).
#[allow(unused_variables)]
pub fn gsm8k_to_text(example: &Gsm8kExample) -> String {
    // BEGIN YOUR CODE
    todo!()
    // END YOUR CODE
}
