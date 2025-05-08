from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from asr.modules import Linear


class AdditiveAttention(nn.Module):
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
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query)+ self.bias)).suqeeze(-1)
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), value)

        context += query

        return context, attn