from typing import Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor

from asr.modules.conv_base import BaseConv2d


class DepthWiseConv2d(BaseConv2d):
    r"""2D depthwise convolution layer.

    Applies a separate 2D convolution filter to each input channel (groups
    equal to ``in_channels``). The number of output channels must be a
    constant multiple of the number of input channels.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels. Must be a multiple of
            ``in_channels``.
        kernel_size (Union[int, Tuple]): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: ``2``.
        padding (int): Zero-padding added to both sides. Default: ``0``.

    Inputs: inputs, input_lengths
        - **inputs** (batch, in_channels, height, width): Input feature tensor.
        - **input_lengths** (batch,): Optional sequence lengths.

    Returns: output (or output, output_lengths)
        - **output** (batch, out_channels, height', width'): Convolved tensor.
        - **output_lengths** (batch,): Adjusted lengths (only when ``input_lengths`` provided).

    Examples::

        >>> conv = DepthWiseConv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2)
        >>> x = torch.randn(2, 256, 10, 100)
        >>> out = conv(x)
        >>> out.shape
        torch.Size([2, 256, 5, 50])
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple],
            stride: int = 2,
            padding: int = 0,
    ) -> None:
        super(DepthWiseConv2d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )

    def forward(self, inputs: Tensor, input_lengths: Optional[Tensor] = None) -> Tensor:
        if input_lengths is None:
            return self.conv(inputs)
        else:
            return self.conv(inputs), self._get_sequence_lengths(input_lengths)
