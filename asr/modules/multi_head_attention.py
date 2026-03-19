from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor

from asr.modules import DotProductAttention, Linear


class MultiHeadAttention(nn.Module):
    r"""Multi-head scaled dot-product attention.

    Splits queries, keys, and values into multiple heads, applies scaled
    dot-product attention independently in each head, and concatenates the
    results.

    Reference:
        "Attention Is All You Need" - Vaswani et al.
        https://arxiv.org/abs/1706.03762

    Args:
        dim (int): Total model dimensionality. Must be divisible by ``num_heads``.
            Default: ``512``.
        num_heads (int): Number of parallel attention heads. Default: ``8``.

    Inputs: query, key, value, mask
        - **query** (batch, time_q, dim): Query tensor.
        - **key** (batch, time_k, dim): Key tensor.
        - **value** (batch, time_k, dim): Value tensor.
        - **mask** (batch, time_q, time_k): Optional boolean mask.

    Returns: context, attn
        - **context** (batch, time_q, dim): Concatenated attention output.
        - **attn** (batch, heads, time_q, time_k): Attention weight distribution.

    Examples::

        >>> mha = MultiHeadAttention(dim=512, num_heads=8)
        >>> q = torch.randn(2, 10, 512)
        >>> ctx, weights = mha(q, q, q)
        >>> ctx.shape
        torch.Size([2, 10, 512])
    """
    def __init__(self, dim: int = 512, num_heads: int = 8) -> None:
        super(MultiHeadAttention, self).__init__()

        assert dim % num_heads == 0, "hidden_dim % num_heads should be zero."

        self.d_head = int(dim / num_heads)
        self.num_heads = num_heads
        self.query_proj = Linear(dim, self.d_head * num_heads)
        self.key_proj = Linear(dim, self.d_head * num_heads)
        self.value_proj = Linear(dim, self.d_head * num_heads)
        self.scaled_dot_attn = DotProductAttention(dim, scale=True)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        context, attn = self.scaled_dot_attn(query, key, value, mask)

        context = context.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.d_head)

        return context, attn
