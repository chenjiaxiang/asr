import torch.nn as nn
from torch import Tensor


class GLU(nn.Module):
    r"""Gated Linear Unit (GLU) activation function.

    Splits the input tensor in half along ``dim`` and applies a sigmoid gate
    to the second half, then multiplies element-wise with the first half:
    ``output = input_a * sigmoid(input_b)``.

    Reference:
        "Language Modeling with Gated Convolutional Networks" - Dauphin et al.
        https://arxiv.org/abs/1612.08083

    Args:
        dim (int): Dimension along which to split the input tensor.

    Inputs: inputs
        - **inputs** (batch, *, 2 * channels, *): Tensor to gate. The size along
          ``dim`` must be even.

    Returns: output
        - **output** (batch, *, channels, *): Gated tensor with half the size
          along ``dim``.

    Examples::

        >>> glu = GLU(dim=1)
        >>> x = torch.randn(2, 64, 10)
        >>> out = glu(x)
        >>> out.shape
        torch.Size([2, 32, 10])
    """
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()
