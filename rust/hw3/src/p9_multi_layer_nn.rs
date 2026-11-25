//! Problem 9: Arbitrary-depth ReLU neural network (`MultiLayerNN`).

#[allow(unused_imports)]
use candle_core::{Device, Result, Tensor, Var};

use crate::{Linear, Module};

/// Deep ReLU neural network with arbitrary hidden layer widths:
///
/// $$h(x) = W_L \, \sigma(W_{L-1} \, \sigma(\dots W_2 \, \sigma(W_1 x) \dots))$$
///
/// where $\sigma$ is ReLU ([`Tensor::relu`]).
///
/// Store all layers in order in the public `linears` field (`Vec<Linear>`, the Rust
/// counterpart to PyTorch's `nn.ModuleList`), with length `hidden_dims.len() + 1`.
/// Apply ReLU after every layer **except** the final output layer.
#[derive(Debug, Clone)]
pub struct MultiLayerNN {
    /// Ordered list of `Linear` layers (`hidden_dims.len() + 1` total).
    pub linears: Vec<Linear>,
}

impl MultiLayerNN {
    /// Initialize a deep ReLU network with the given input, output, and hidden dimensions.
    ///
    /// # Inputs
    /// - `in_dim`: input feature dimension
    /// - `out_dim`: output feature dimension
    /// - `hidden_dims`: slice of hidden layer widths in order (may be empty for a single layer)
    /// - `device`: Candle device on which to allocate layer weights
    #[allow(unused_variables)]
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        hidden_dims: &[usize],
        device: &Device,
    ) -> Result<Self> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }
}

impl Module for MultiLayerNN {
    /// Apply all hidden linear layers with ReLU activations, then the final linear output layer.
    #[allow(unused_variables)]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }

    /// Collect trainable parameters from every `Linear` layer in `self.linears`.
    fn parameters(&self) -> Vec<Var> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }
}
