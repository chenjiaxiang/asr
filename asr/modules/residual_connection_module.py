from typing import Optional

import torch.nn as nn
from torch import Tensor


class ResidualConnectionModule(nn.Module):
    r"""Residual connection wrapper with optional scaling factors.

    Computes ``(module(inputs) * module_factor) + (inputs * input_factor)``.
    When ``mask`` is provided, it is forwarded to the wrapped ``module``.

    Args:
        module (nn.Module): The sub-module to wrap with a residual connection.
        module_factor (float): Scale applied to the module output before adding
            the residual. Default: ``1.0``.
        input_factor (float): Scale applied to the residual (input) before
            adding. Default: ``1.0``.

    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Input tensor.
        - **mask** (batch, time, time): Optional attention mask tensor.

    Returns: output
        - **output** (batch, time, dim): Output tensor after residual addition.

    Examples::

        >>> ff = FeedForwardModule(encoder_dim=512)
        >>> residual = ResidualConnectionModule(ff, module_factor=0.5)
        >>> x = torch.randn(2, 10, 512)
        >>> out = residual(x)
        >>> out.shape
        torch.Size([2, 10, 512])
    """
    def __init__(
            self,
            module: nn.Module,
            module_factor: float = 1.0,
            input_factor: float = 1.0,
    ) -> None:
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is None:
            return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)
        else:
            return (self.module(inputs, mask) * self.module_factor) + (inputs * self.input_factor)
