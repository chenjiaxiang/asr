from typing import Tuple
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor

class Linear(nn.Module):
    r"""Wrapper class of ``torch.nn.Linear``.

    Applies a linear transformation to the input with Xavier normal initialization.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If set to ``False``, the layer will not learn an additive bias. Default: ``True``.

    Inputs: x
        - **x** (batch, *, in_features): Tensor containing input features.

    Returns: output
        - **output** (batch, *, out_features): Tensor containing output features.

    Examples::

        >>> linear = Linear(512, 256)
        >>> x = torch.randn(2, 10, 512)
        >>> out = linear(x)
        >>> out.shape
        torch.Size([2, 10, 256])
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

class View(nn.Module):
    r"""Wrapper class of ``torch.Tensor.view``.

    Reshapes the input tensor to the specified shape.

    Args:
        shape (Tuple): Desired output shape.
        contiguous (bool): If ``True``, make tensor contiguous before reshaping. Default: ``False``.

    Inputs: inputs
        - **inputs** (batch, *): Tensor to reshape.

    Returns: output
        - **output** (batch, *shape): Reshaped tensor.

    Examples::

        >>> view = View(shape=(-1, 512))
        >>> x = torch.randn(2, 10, 512)
        >>> out = view(x)
        >>> out.shape
        torch.Size([20, 512])
    """
    def __init__(self, shape: Tuple, contiguous: bool = False) -> None:
        super(View, self).__init__()
        self.shape = shape
        self.contiguous = contiguous

    def forward(self, inputs: Tensor) -> Tensor:
        if self.contiguous:
            inputs = inputs.contiguous()
        return inputs.view(*self.shape)

class Transpose(nn.Module):
    r"""Wrapper class of ``torch.Tensor.transpose``.

    Transposes the specified dimensions of the input tensor.

    Args:
        shape (Tuple): A tuple of two dimension indices to swap.

    Inputs: inputs
        - **inputs** (batch, *): Tensor to transpose.

    Returns: output
        - **output** (batch, *): Transposed tensor.

    Examples::

        >>> transpose = Transpose(shape=(1, 2))
        >>> x = torch.randn(2, 10, 512)
        >>> out = transpose(x)
        >>> out.shape
        torch.Size([2, 512, 10])
    """
    def __init__(self, shape: Tuple) -> None:
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.transpose(*self.shape)
