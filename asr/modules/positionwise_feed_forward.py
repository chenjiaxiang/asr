import torch.nn as nn
from torch import Tensor

from asr.modules.wrapper import Linear


class PositionwiseFeedForward(nn.Module):
    r"""Position-wise feed-forward network used in transformer models.

    Applies two linear transformations with a ReLU activation and dropout
    in between. The same transformation is applied to each position
    independently.

    Reference:
        "Attention Is All You Need" - Vaswani et al.
        https://arxiv.org/abs/1706.03762

    Args:
        d_model (int): Model dimensionality (input and output size). Default: ``512``.
        d_ff (int): Inner layer dimensionality. Default: ``2048``.
        dropout_p (float): Dropout probability. Default: ``0.3``.

    Inputs: inputs
        - **inputs** (batch, time, d_model): Input tensor.

    Returns: output
        - **output** (batch, time, d_model): Output tensor after feed-forward transformation.

    Examples::

        >>> ff = PositionwiseFeedForward(d_model=512, d_ff=2048)
        >>> x = torch.randn(2, 10, 512)
        >>> out = ff(x)
        >>> out.shape
        torch.Size([2, 10, 512])
    """
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout_p: float = 0.3) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.feed_forward = nn.Sequential(
            Linear(d_model, d_ff),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            Linear(d_ff, d_model),
            nn.Dropout(dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.feed_forward(inputs)
