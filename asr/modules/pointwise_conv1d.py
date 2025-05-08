import torch.nn as nn
from torch import Tensor

from asr.modules.conv_base import BaseConv1d

class PointWiseConv1d(BaseConv1d):
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