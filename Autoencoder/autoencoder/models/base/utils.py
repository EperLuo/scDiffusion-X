from typing import Callable, List, Optional
import torch

def unsqueeze_right(x, num_dims=1):
    """
    Unsqueezes the last `num_dims` dimensions of `x`.

    Args:
        x (torch.Tensor): Input tensor.
        num_dims (int, optional): Number of dimensions to unsqueeze. Defaults to 1.

    Returns:
        torch.Tensor: Tensor with unsqueezed dimensions.
    """
    return x.view(x.shape + (1,) * num_dims)

def pad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.

    Parameters
    ----------
    x : Tensor, shape (bs, *dim)
        represents the source minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    t : Tensor, shape (bs, number of x dimensions)

    Example
    -------
    x: Tensor (bs, C, W, H)
    t: Vector (bs)
    pad_t_like_x(t, x): Tensor (bs, 1, 1, 1)
    """
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))

def kl_std_normal(mean_squared, var):
    """
    Computes Gaussian KL divergence.

    Args:
        mean_squared (torch.Tensor): Mean squared values.
        var (torch.Tensor): Variance values.

    Returns:
        torch.Tensor: Gaussian KL divergence.
    """
    return 0.5 * (var + mean_squared - torch.log(var.clamp(min=1e-15)) - 1.0)

class MLP(torch.nn.Module):
    def __init__(self, 
                 dims: List[int],
                 norm: bool, 
                 dropout: bool, 
                 dropout_p: float, 
                 activation: Optional[Callable] = torch.nn.ELU, 
                 final_activation: Optional[str] = None,
                 norm_type: Optional[str] = 'batchnorm'):
        """
        Multi-Layer Perceptron (MLP) model.

        Args:
            dims (List[int]): List of dimensions for each layer.
            batch_norm (bool): Whether to use batch normalization.
            dropout (bool): Whether to use dropout.
            dropout_p (float): Dropout probability.
            activation (Optional[Callable], optional): Activation function. Defaults to torch.nn.SELU.
            final_activation (Optional[str], optional): Final activation function ("tanh", "sigmoid", or None). Defaults to None.
        """
        super(MLP, self).__init__()

        # Attributes 
        self.dims = dims
        self.norm = norm
        self.activation = activation

        # MLP 
        layers = []
        for i in range(len(self.dims[:-2])):
            block = []
            block.append(torch.nn.Linear(self.dims[i], self.dims[i+1]))
            if norm: 
                if norm_type == 'batchnorm':
                    block.append(torch.nn.BatchNorm1d(self.dims[i+1]))
                else:
                    block.append(torch.nn.LayerNorm(self.dims[i+1]))
            block.append(self.activation())
            if dropout:
                block.append(torch.nn.Dropout(dropout_p))
            layers.append(torch.nn.Sequential(*block))
        
        # Last layer without activation 
        layers.append(torch.nn.Linear(self.dims[-2], self.dims[-1]))
        # Compile the neural net
        self.net = torch.nn.Sequential(*layers)
        
        if final_activation == "tanh":
            self.final_activation = torch.nn.Tanh()
        elif final_activation == "sigmoid":
            self.final_activation = torch.nn.Sigmoid()
        else:
            self.final_activation = None

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.net(x)
        if not self.final_activation:
            return x
        else:
            return self.final_activation(x)
