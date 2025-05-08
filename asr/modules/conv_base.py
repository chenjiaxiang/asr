import torch.nn as nn
from torch import Tensor

class BaseConv1d(nn.Module):
    def __init__(self) -> None:
        super(BaseConv1d, self).__init__()

    def _get_sequence_lengths(self, seq_lengths: Tensor) -> Tensor:
        return (
            seq_lengths + 2 * self.conv.padding[0] - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) -1
        ) // self.conv.stride[0]  + 1
    
    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

class BaseConv2d(nn.Module):
    def __init__(self) -> None:
        super(BaseConv2d, self).__init__()

    def _get_sequence_lengths(self, seq_lengths: Tensor) -> Tensor:
        return (
            seq_lengths + 2 * self.conv.padding[0] - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - 1
        ) // self.conv.stride[0] + 1
    
    def forward(self, *args, **kwargs) -> None:
        raise NotImplementedError