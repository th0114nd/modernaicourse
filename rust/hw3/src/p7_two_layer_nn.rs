//! Problem 7: Two-layer ReLU neural network (`TwoLayerNN`).

use candle_core::{bail, Device, Result, Tensor, Var};

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
    pub fn new(in_dim: usize, hidden_dim: usize, out_dim: usize, device: &Device) -> Result<Self> {
        // BEGIN YOUR CODE
        if in_dim == 0 || hidden_dim == 0 || out_dim == 0 {
            bail!("degenerate dims");
        }
        let linear1 = Linear::new(in_dim, hidden_dim, device)?;
        let linear2 = Linear::new(hidden_dim, out_dim, device)?;
        Ok(Self { linear1, linear2 })
        // END YOUR CODE
    }
}

impl Module for TwoLayerNN {
    /// Apply `linear1`, ReLU, then `linear2` to an input tensor of shape `[..., in_dim]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // BEGIN YOUR CODE
        let h1 = self.linear1.forward(x)?;
        let h2 = h1.relu()?;
        let h3 = self.linear2.forward(&h2)?;
        Ok(h3)
        // END YOUR CODE
    }

    /// Return all trainable parameters from `linear1` and `linear2`.
    fn parameters(&self) -> Vec<Var> {
        // BEGIN YOUR CODE
        let mut params = self.linear1.parameters();
        params.extend(self.linear2.parameters());
        params
        // END YOUR CODE
    }
}
