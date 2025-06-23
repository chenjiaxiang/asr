from typing import Tuple

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from asr.modules import Linear

class LocationAwareAttention(nn.Module):
    def __init__(self, dim: int = 1024, attn_dim: int = 1024, smoothing: bool = False) -> None:
        super(LocationAwareAttention, self).__init__()
        self.location_conv = nn.Conv1d(in_channels=1, out_channels=attn_dim, kernel_size=3, padding=1)
        self.query_proj = Linear(dim, attn_dim, bias=False)
        self.value_proj = Linear(dim, attn_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(attn_dim).uniform_(-0.1, 0.1))
        self.fc = Linear(attn_dim, 1, bias=True)
        self.smoothing = smoothing

    def forward(
            self,
            query: Tensor,
            value: Tensor,
            last_alignment_energy: Tensor
    ) -> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, seq_length = query.size(0), query.size(2), value.size(1)

        if last_alignment_energy is None:
            last_alignment_energy = value.new_zeros(batch_size, seq_length) 

        last_alignment_energy = self.location_conv(last_alignment_energy.unsqueeze(dim=1))
        last_alignment_energy = last_alignment_energy.transpose(1, 2)

        alignment_energy = self.fc(
            torch.tanh(self.query_proj(query) + self.value_proj(value) + last_alignment_energy + self.bias)
        ).squeeze(dim=-1)

        if self.smoothing:
            alignment_energy = torch.sigmoid(alignment_energy)
            alignment_energy = torch.div(alignment_energy, alignment_energy.sum(dim=-1).unsqueeze(dim=-1))
        else:
            alignment_energy = F.softmax(alignment_energy, dim=-1)

        context = torch.bmm(alignment_energy.unsqueeze(dim=1), value)

        return context, alignment_energy