//! Problem 9: Arbitrary-depth ReLU neural network (`MultiLayerNN`).

#[allow(unused_imports)]
use candle_core::{bail, Device, Error, Result, Tensor, Var};

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
        let mut dims = vec![in_dim];
        dims.extend(hidden_dims);
        dims.push(out_dim);
        for dim in dims.iter().copied() {
            if dim == 0 {
                bail!("degenerate dim");
            }
        }
        Ok(Self {
            linears: dims
                .windows(2)
                .map(|ds| Linear::new(ds[0], ds[1], device))
                .collect::<Result<Vec<_>>>()?,
        })
        // END YOUR CODE
    }
}

impl Module for MultiLayerNN {
    /// Apply all hidden linear layers with ReLU activations, then the final linear output layer.
    #[allow(unused_variables)]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // BEGIN YOUR CODE
        let mut h = x.clone();
        let num_layers = self.linears.len();
        let (last, early) = self
            .linears
            .split_last()
            .ok_or(Error::msg("need at least one layer"))?;
        for layer in early {
            h = layer.forward(&h)?;
            h = h.relu()?;
        }
        last.forward(&h)
        // END YOUR CODE
    }

    /// Collect trainable parameters from every `Linear` layer in `self.linears`.
    fn parameters(&self) -> Vec<Var> {
        // BEGIN YOUR CODE
        self.linears.iter().flat_map(Module::parameters).collect()
        // END YOUR CODE
    }
}
