import torch.nn as nn
from torch import Tensor

from asr.modules.swish import Swish
from asr.modules.wrapper import Linear


class FeedForwardModule(nn.Module):
    r"""Feed-forward module used in Conformer encoder blocks.

    Applies layer normalization followed by two linear projections with a
    Swish activation and dropout in between, as described in the Conformer paper.

    Reference:
        "Conformer: Convolution-augmented Transformer for Speech Recognition"
        - Gulati et al.
        https://arxiv.org/abs/2005.08100

    Args:
        encoder_dim (int): Model dimensionality (input and output size). Default: ``512``.
        expansion_factor (int): Inner layer is ``encoder_dim * expansion_factor``. Default: ``4``.
        dropout_p (float): Dropout probability applied after each linear layer. Default: ``0.1``.

    Inputs: inputs
        - **inputs** (batch, time, encoder_dim): Input tensor.

    Returns: output
        - **output** (batch, time, encoder_dim): Output tensor.

    Examples::

        >>> ff = FeedForwardModule(encoder_dim=512, expansion_factor=4)
        >>> x = torch.randn(2, 10, 512)
        >>> out = ff(x)
        >>> out.shape
        torch.Size([2, 10, 512])
    """
    def __init__(
            self,
            encoder_dim: int = 512,
            expansion_factor: int = 4,
            dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)
