from typing import Optional

import torch.nn as nn
from torch import Tensor

class ResidualConnectionModule(nn.Module):
    def __init__(
            self,
            module: nn.Module,
            module_factor: float = 1.0,
            input_factor: float = 1.0,
        )  -> None:
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is None:
            return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)
        else:
            return (self.module(inputs, mask) * self.module_factor) + (inputs * self.input_factor)