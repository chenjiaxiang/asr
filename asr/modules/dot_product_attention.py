from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DotProductAttention(nn.Module):
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
        