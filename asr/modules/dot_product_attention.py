from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DotProductAttention(nn.Module):
    r"""Scaled or unscaled dot-product attention.

    Computes attention weights via dot product of queries and keys, optionally
    scaled by ``1 / sqrt(dim)``, then applies them to values.

    Reference:
        "Attention Is All You Need" - Vaswani et al.
        https://arxiv.org/abs/1706.03762

    Args:
        dim (int): Dimensionality used for the scaling factor.
        scale (bool): If ``True``, divide scores by ``sqrt(dim)``. Default: ``True``.

    Inputs: query, key, value, mask
        - **query** (batch, time_q, dim) or (batch, heads, time_q, d_head): Query tensor.
        - **key** (batch, time_k, dim) or (batch, heads, time_k, d_head): Key tensor.
        - **value** (batch, time_k, dim) or (batch, heads, time_k, d_head): Value tensor.
        - **mask** (batch, time_q, time_k): Optional boolean mask; masked positions
          are filled with ``-1e4`` before softmax.

    Returns: context, attn
        - **context** (batch, time_q, dim): Context vector (weighted sum of values).
        - **attn** (batch, time_q, time_k): Attention weight distribution.

    Examples::

        >>> attn = DotProductAttention(dim=512, scale=True)
        >>> q = torch.randn(2, 10, 512)
        >>> k = torch.randn(2, 20, 512)
        >>> v = torch.randn(2, 20, 512)
        >>> ctx, weights = attn(q, k, v)
        >>> ctx.shape
        torch.Size([2, 10, 512])
    """
    def __init__(self, dim: int, scale: bool = True) -> None:
        super(DotProductAttention, self).__init__()
        if scale:
            self.sqrt_dim = np.sqrt(dim)
        else:
            self.sqrt_dim = 1

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if len(query.size()) == 3:
            score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        else:
            score = torch.matmul(query, key.transpose(2, 3)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask, -1e4)

        attn = F.softmax(score, -1)

        if len(query.size()) == 3:
            context = torch.bmm(attn, value)
        else:
            context = torch.matmul(attn, value)

        return context, attn
