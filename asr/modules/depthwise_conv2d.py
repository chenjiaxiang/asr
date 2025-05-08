from typing import Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor

from asr.modules.conv_base import BaseConv2d

class DepthWiseConv2d(BaseConv2d):
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