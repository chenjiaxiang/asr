from typing import Optional

import torch.nn as nn
from torch import Tensor

from asr.modules.positional_encoding import RelPositionalEncoding
from asr.modules.relative_multi_head_attention import RelativeMultiHeadAttention


class MultiHeadedSelfAttentionModule(nn.Module):
    r"""Multi-headed self-attention module used in Conformer encoder blocks.

    Applies layer normalization, relative positional encoding, and multi-headed
    self-attention with relative positional biases, followed by dropout.

    Reference:
        "Conformer: Convolution-augmented Transformer for Speech Recognition"
        - Gulati et al.
        https://arxiv.org/abs/2005.08100

    Args:
        d_model (int): Model dimensionality.
        num_heads (int): Number of attention heads.
        dropout_p (float): Dropout probability. Default: ``0.1``.

    Inputs: inputs, mask
        - **inputs** (batch, time, d_model): Input tensor.
        - **mask** (batch, time, time): Optional boolean attention mask.

    Returns: output
        - **output** (batch, time, d_model): Attention output after dropout.

    Examples::

        >>> mhsa = MultiHeadedSelfAttentionModule(d_model=512, num_heads=8)
        >>> x = torch.randn(2, 10, 512)
        >>> out = mhsa(x)
        >>> out.shape
        torch.Size([2, 10, 512])
    """
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            dropout_p: float = 0.1,
    ) -> None:
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.postional_encoding = RelPositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size = inputs.size(0)
        pos_embedding = self.postional_encoding(inputs)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)

        return self.dropout(outputs)
