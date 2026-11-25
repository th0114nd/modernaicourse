//! Problem 7: Two-layer ReLU neural network (`TwoLayerNN`).

#[allow(unused_imports)]
use candle_core::{Device, Result, Tensor, Var};

use crate::{Linear, Module};

/// Two-layer neural network with ReLU hidden activation:
///
/// $$h(x) = W_2 \, \sigma(W_1 x)$$
///
/// where $\sigma(z) = \max(0, z)$ is ReLU ([`Tensor::relu`]).
///
/// Store the two sublayers as public [`Linear`] fields `linear1` (`in_dim -> hidden_dim`)
/// and `linear2` (`hidden_dim -> out_dim`).
#[derive(Debug, Clone)]
pub struct TwoLayerNN {
    /// First linear layer (`in_dim -> hidden_dim`).
    pub linear1: Linear,
    /// Second linear layer (`hidden_dim -> out_dim`).
    pub linear2: Linear,
}

impl TwoLayerNN {
    /// Initialize a two-layer ReLU neural network.
    ///
    /// # Inputs
    /// - `in_dim`: input feature dimension
    /// - `hidden_dim`: hidden layer width
    /// - `out_dim`: output feature dimension
    /// - `device`: Candle device on which to allocate layer weights
    #[allow(unused_variables)]
    pub fn new(in_dim: usize, hidden_dim: usize, out_dim: usize, device: &Device) -> Result<Self> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }
}

impl Module for TwoLayerNN {
    /// Apply `linear1`, ReLU, then `linear2` to an input tensor of shape `[..., in_dim]`.
    #[allow(unused_variables)]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }

    /// Return all trainable parameters from `linear1` and `linear2`.
    fn parameters(&self) -> Vec<Var> {
        // BEGIN YOUR CODE
        todo!()
        // END YOUR CODE
    }
}
