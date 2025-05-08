from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from asr.modules import Conv2dExtractor


class DeepSpeed2Extractor(Conv2dExtractor):
    def __init__(
            self,
            input_dim: int, 
            in_channels: int = 1,
            out_channels: int = 32,
            activations: str = "hardtanh"
    ) -> None:
        super(DeepSpeed2Extractor, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        from asr.modules import MaskConv2d

        self.conv = MaskConv2d(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5), bias=False),
                nn.BatchNorm2d(out_channels),
                self.activation,
                nn.Conv2d(out_channels, out_channels, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5), bias=False),
                nn.BatchNorm2d(out_channels),
                self.activation,
            )
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        return super().forward(inputs, input_lengths)