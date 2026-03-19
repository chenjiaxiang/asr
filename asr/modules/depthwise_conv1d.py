from typing import Optional

import torch.nn as nn
from torch import Tensor

from asr.modules.conv_base import BaseConv1d


class DepthWiseConv1d(BaseConv1d):
    r"""1D depthwise convolution layer.

    Applies a separate convolution filter to each input channel (groups equal
    to ``in_channels``). The number of output channels must be a constant
    multiple of the number of input channels.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels. Must be a multiple of
            ``in_channels``.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: ``1``.
        padding (int): Zero-padding added to both sides of the input. Default: ``0``.
        bias (bool): If ``True``, add a learnable bias. Default: ``False``.

    Inputs: inputs, input_lengths
        - **inputs** (batch, in_channels, time): Input feature tensor.
        - **input_lengths** (batch,): Optional sequence lengths.

    Returns: output (or output, output_lengths)
        - **output** (batch, out_channels, time'): Convolved tensor.
        - **output_lengths** (batch,): Adjusted lengths (only when ``input_lengths`` provided).

    Examples::

        >>> conv = DepthWiseConv1d(in_channels=256, out_channels=256, kernel_size=31, padding=15)
        >>> x = torch.randn(2, 256, 100)
        >>> out = conv(x)
        >>> out.shape
        torch.Size([2, 256, 100])
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False,
    ) -> None:
        super(DepthWiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor, input_lengths: Optional[Tensor] = None) -> Tensor:
        if input_lengths is None:
            return self.conv(inputs)
        else:
            return self.conv(inputs), self._get_sequence_lengths(input_lengths)
