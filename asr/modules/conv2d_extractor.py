import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from asr.modules.swish import Swish
from asr.utils import get_class_name


class Conv2dExtractor(nn.Module):
    supported_activations = {
        "hardtanh": nn.Hardtanh(0, 20, inplace=True),
        "relu": nn.ReLU(inplace=True),
        "elu": nn.ELU(inplace=True),
        "leaky_relu": nn.LeakyReLU(inplace=True),
        "gelu": nn.GELU(),
        "swish": Swish(),
    }

    def __init__(self, input_dim: int, activation: str = "hardtanh") -> None:
        super(Conv2dExtractor, self).__init__()
        self.input_dim = input_dim
        self.activation = Conv2dExtractor.supported_activations[activation]
        self.conv = None

    def get_output_lengths(self, seq_lengths: Tensor) -> int:
        assert self.conv is not None, "self.conv should be defined"

        for module in self.conv:
            if isinstance(module, nn.Conv2d):
                numerator = seq_lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1
                seq_lengths = numerator.float() / float(module.stride[1])
                seq_lengths = seq_lengths.int() + 1

            elif isinstance(module, nn.MaxPool2d):
                seq_lengths >>= 1
        
        return seq_lengths.int()

    def get_output_dim(self) -> int:
        if get_class_name(self)  == "VGGExtractor":
            output_dim = (self.input_dim -1) << 5 if self.input_dim % 2 else self.input_dim << 5
        
        elif get_class_name(self) == "DeepSpeed2Extractor":
            output_dim = int(math.floor(self.input_dim + 2 * 20 -  41) / 2 + 1)
            output_dim = int(math.floor(output_dim + 2 * 10 - 21) / 2 + 1)
            output_dim << 5
        
        elif get_class_name(self) == "Conv2dSubSampling":
            factor = ((self.input_dim - 1) // 2 - 1) // 2
            output_dim = self.out_channels * factor

        else:
            raise ValueError(f"Unsupported Extractor : {self.extractor}")

        return output_dim

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        outputs, output_lengths = self.conv(inputs.unsqueeze(1).transpose(2, 3), input_lengths)

        batch_size, channels, dimension, seq_lengths = outputs.size()
        outputs = outputs.permute(0, 3, 1, 2)
        outputs = outputs.view(batch_size, seq_lengths, channels * dimension)

        return outputs, output_lengths