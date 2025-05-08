from typing import Tuple

import torch 
import torch.nn as nn
from torch import Tensor

from asr.modules.conv2d_extractor import Conv2dExtractor
from asr.modules.depthwise_conv2d import DepthWiseConv2d

class Conv2dSubSampling(Conv2dExtractor):
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