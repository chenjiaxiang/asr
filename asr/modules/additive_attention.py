from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from asr.modules import Linear


class AdditiveAttention(nn.Module):
    r"""Additive (Bahdanau) attention mechanism.

    Computes attention scores as a learnable weighted sum of the query and key
    projections passed through a ``tanh`` non-linearity, then reduces to a
    scalar score with a final linear projection.

    Reference:
        "Neural Machine Translation by Jointly Learning to Align and Translate"
        - Bahdanau et al.
        https://arxiv.org/abs/1409.0473

    Args:
        dim (int): Dimensionality of query, key, and value tensors.

    Inputs: query, key, value
        - **query** (batch, time_q, dim): Query tensor.
        - **key** (batch, time_k, dim): Key tensor.
        - **value** (batch, time_k, dim): Value tensor.

    Returns: context, attn
        - **context** (batch, time_q, dim): Context vector (weighted sum of values)
          added to the query.
        - **attn** (batch, time_k): Attention weight distribution.

    Examples::

        >>> attn = AdditiveAttention(dim=512)
        >>> q = torch.randn(2, 1, 512)
        >>> k = torch.randn(2, 20, 512)
        >>> v = torch.randn(2, 20, 512)
        >>> ctx, weights = attn(q, k, v)
        >>> ctx.shape
        torch.Size([2, 1, 512])
    """
    def __init__(self, dim: int) -> None:
        super(AdditiveAttention, self).__init__()
        self.query_proj = Linear(dim, dim, bias=False)
        self.key_proj = Linear(dim, dim, bias=False)
        self.score_proj = Linear(dim, 1)
        self.bias = nn.Parameter(torch.rand(dim).uniform_(-0.1, 0.1))

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor
    ) -> Tuple[Tensor, Tensor]:
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), value)

        context += query

        return context, attn
