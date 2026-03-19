from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from asr.modules.conv2d_extractor import Conv2dExtractor
from asr.modules.depthwise_conv2d import DepthWiseConv2d


class Conv2dSubSampling(Conv2dExtractor):
    r"""2D convolutional sub-sampling feature extractor.

    Applies two consecutive 2D convolutions (each with kernel size 3 and
    stride 2) to down-sample the time and frequency dimensions by a factor
    of 4. Used to reduce the length of the feature sequence before feeding
    into the transformer encoder.

    Args:
        input_dim (int): Number of input frequency bins.
        in_channels (int): Number of input channels (typically ``1`` for raw
            spectrogram).
        out_channels (int): Number of output channels after convolution.
        activation (str): Activation function name. Default: ``"relu"``.

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, input_dim): Input feature tensor.
        - **input_lengths** (batch,): Length of each input sequence.

    Returns: outputs, output_lengths
        - **outputs** (batch, time', out_channels * freq'): Sub-sampled features.
        - **output_lengths** (batch,): Lengths after sub-sampling.

    Examples::

        >>> extractor = Conv2dSubSampling(input_dim=80, in_channels=1, out_channels=256)
        >>> x = torch.randn(2, 100, 80)
        >>> lengths = torch.tensor([100, 80])
        >>> out, out_len = extractor(x, lengths)
    """
    def __init__(
            self,
            input_dim: int,
            in_channels: int,
            out_channels: int,
            activation: str = "relu",
    ) -> None:
        super(Conv2dSubSampling, self).__init__(input_dim, activation)
        self.in_channles = in_channels
        self.out_channels = out_channels

        from asr.modules import MaskConv2d

        self.conv = MaskConv2d(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
                self.activation,
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
                self.activation,
            )
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        outputs, output_lengths = super().forward(inputs, input_lengths)

        return outputs, output_lengths


class DepthWiseConv2dSubSampling(Conv2dExtractor):
    r"""2D depthwise convolutional sub-sampling feature extractor.

    Similar to ``Conv2dSubSampling`` but uses a depthwise convolution in the
    second layer to reduce the number of parameters.

    Args:
        input_dim (int): Number of input frequency bins.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (str): Activation function name. Default: ``"relu"``.

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, input_dim): Input feature tensor.
        - **input_lengths** (batch,): Length of each input sequence.

    Returns: outputs, output_lengths
        - **outputs** (batch, time', out_channels * freq'): Sub-sampled features.
        - **output_lengths** (batch,): Lengths after sub-sampling.
    """
    def __init__(
            self,
            input_dim: int,
            in_channels: int,
            out_channels: int,
            activation: str = "relu",
    ) -> None:
        super(DepthWiseConv2dSubSampling, self).__init__(input_dim, activation)
        self.in_channels = in_channels
        self.out_channels = out_channels

        from asr.modules import MaskConv2d

        self.conv = MaskConv2d(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
                self.activation,
                DepthWiseConv2d(out_channels, out_channels, kernel_size=3, stride=2),
                self.activation,
            )
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        outputs, output_lengths = super().forward(inputs, input_lengths)

        return outputs, output_lengths
