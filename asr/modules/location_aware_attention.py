from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from asr.modules import Linear


class LocationAwareAttention(nn.Module):
    r"""Location-aware (hybrid) attention mechanism.

    Extends additive attention by incorporating information about the previous
    alignment energy through a convolution-based location feature extractor.
    This helps the model attend to the correct position when aligning speech
    frames to output tokens.

    Reference:
        "Attention-Based Models for Speech Recognition" - Chorowski et al.
        https://arxiv.org/abs/1506.07503

    Args:
        dim (int): Dimensionality of the query and value tensors. Default: ``1024``.
        attn_dim (int): Dimensionality of the attention projection space. Default: ``1024``.
        smoothing (bool): If ``True``, use sigmoid + normalization instead of softmax
            for the alignment energy. Default: ``False``.

    Inputs: query, value, last_alignment_energy
        - **query** (batch, 1, dim): Current decoder hidden state.
        - **value** (batch, time, dim): Encoder output sequence.
        - **last_alignment_energy** (batch, time): Previous alignment energies,
          or ``None`` on the first step.

    Returns: context, alignment_energy
        - **context** (batch, 1, dim): Context vector (weighted sum of encoder outputs).
        - **alignment_energy** (batch, time): New alignment energy distribution.

    Examples::

        >>> attn = LocationAwareAttention(dim=512, attn_dim=512)
        >>> q = torch.randn(2, 1, 512)
        >>> v = torch.randn(2, 20, 512)
        >>> ctx, energy = attn(q, v, None)
        >>> ctx.shape
        torch.Size([2, 1, 512])
    """
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
            last_alignment_energy: Optional[Tensor]
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
