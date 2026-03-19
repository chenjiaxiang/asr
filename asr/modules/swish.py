import torch.nn as nn
from torch import Tensor


class Swish(nn.Module):
    r"""Swish activation function.

    Applies the Swish activation: ``output = input * sigmoid(input)``.
    Swish is a self-gated activation function that has been shown to outperform
    ReLU on many deep learning tasks.

    Reference:
        "Searching for Activation Functions" - Ramachandran et al.
        https://arxiv.org/abs/1710.05941

    Inputs: inputs
        - **inputs** (batch, *): Tensor of any shape.

    Returns: output
        - **output** (batch, *): Tensor of the same shape as ``inputs``.

    Examples::

        >>> swish = Swish()
        >>> x = torch.randn(2, 10, 512)
        >>> out = swish(x)
        >>> out.shape
        torch.Size([2, 10, 512])
    """
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()
