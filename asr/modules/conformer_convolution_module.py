import torch.nn as nn
from torch import Tensor

from asr.modules.depthwise_conv1d import DepthWiseConv1d
from asr.modules.glu import GLU
from asr.modules.pointwise_conv1d import PointWiseConv1d
from asr.modules.swish import Swish
from asr.modules.wrapper import Transpose


class ConformerConvModule(nn.Module):
    r"""Convolution module used in Conformer encoder blocks.

    Applies layer normalization followed by a pointwise convolution, a GLU,
    a depthwise convolution, batch normalization, a Swish activation, another
    pointwise convolution, and dropout. Input is transposed for 1D convolution
    and transposed back.

    Reference:
        "Conformer: Convolution-augmented Transformer for Speech Recognition"
        - Gulati et al.
        https://arxiv.org/abs/2005.08100

    Args:
        in_channels (int): Number of input (and output) feature channels.
        kernel_size (int): Kernel size of the depthwise convolution. Must be odd.
            Default: ``31``.
        expansion_factor (int): Expansion factor for the pointwise convolution.
            Currently only ``2`` is supported. Default: ``2``.
        dropout_p (float): Dropout probability. Default: ``0.1``.

    Inputs: inputs
        - **inputs** (batch, time, in_channels): Input tensor.

    Returns: output
        - **output** (batch, time, in_channels): Output tensor.

    Examples::

        >>> conv = ConformerConvModule(in_channels=512, kernel_size=31)
        >>> x = torch.randn(2, 10, 512)
        >>> out = conv(x)
        >>> out.shape
        torch.Size([2, 10, 512])
    """
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 31,
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
    ) -> None:
        super(ConformerConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),
            Transpose(shape=(1, 2)),
            PointWiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True),
            GLU(dim=1),
            DepthWiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(in_channels),
            Swish(),
            PointWiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs).transpose(1, 2)
