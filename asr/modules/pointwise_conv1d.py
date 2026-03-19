import torch.nn as nn
from torch import Tensor

from asr.modules.conv_base import BaseConv1d


class PointWiseConv1d(BaseConv1d):
    r"""1D pointwise (1x1) convolution layer.

    Applies a convolution with ``kernel_size=1``, which is equivalent to a
    linear transformation applied independently at each time step. Commonly
    used to change the number of channels without mixing temporal information.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride of the convolution. Default: ``1``.
        padding (int): Zero-padding added to both sides. Default: ``0``.
        bias (bool): If ``True``, add a learnable bias. Default: ``True``.

    Inputs: inputs
        - **inputs** (batch, in_channels, time): Input feature tensor.

    Returns: output
        - **output** (batch, out_channels, time): Convolved tensor.

    Examples::

        >>> conv = PointWiseConv1d(in_channels=512, out_channels=1024)
        >>> x = torch.randn(2, 512, 100)
        >>> out = conv(x)
        >>> out.shape
        torch.Size([2, 1024, 100])
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True,
    ) -> None:
        super(PointWiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)
