from typing import Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from asr.modules import Conv2dExtractor

class VGGExtractor(Conv2dExtractor):
    def __init__(
        self,
        input_dim: int,
        in_channels: int = 1,
        out_channels: Union[int, Tuple] = (64, 128),
        activation: str = "hardtanh",
    ) -> None:
        super(VGGExtractor, self).__init__(input_dim=input_dim, activation=activation)
        self.in_channels = in_channels
        self.out_channels = out_channels

        from asr.modules import MaskConv2d

        self.conv = MaskConv2d(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels[0]),
                self.activation,
                nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels[0]),
                self.activation,
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels[1]),
                self.activation,
                nn.Conv2d(out_channels[1], out_channels[1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels[1]),
                self.activation,
                nn.MaxPool2d(2, stride=2),
            )
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        return super().forward(inputs, input_lengths)