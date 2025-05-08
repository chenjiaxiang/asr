from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor

from asr.modules import DotProductAttention, Linear


class MultiHeadAttention(nn.Module):
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
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).transpost(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        
        context, attn = self.scaled_dot_attn(query, key, value, mask)

        context = context.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.d_head)

        return context, attn