import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    r"""Sinusoidal positional encoding for transformer models.

    Adds fixed sinusoidal positional encodings to token embeddings so the model
    can make use of the order of the sequence. The encoding for position ``pos``
    and dimension ``i`` is:

    - ``PE(pos, 2i) = sin(pos / 10000^(2i/d_model))``
    - ``PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))``

    Args:
        d_model (int): Dimensionality of the model embeddings. Default: ``512``.
        max_len (int): Maximum sequence length for precomputed encodings. Default: ``5000``.

    Inputs: length
        - **length** (int): Number of time steps to return.

    Returns: pe
        - **pe** (1, length, d_model): Positional encoding slice.

    Examples::

        >>> pe = PositionalEncoding(d_model=512)
        >>> enc = pe(100)
        >>> enc.shape
        torch.Size([1, 100, 512])
    """
    def __init__(
            self,
            d_model: int = 512,
            max_len: int = 5000
    ) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length]


class RelPositionalEncoding(nn.Module):
    r"""Relative positional encoding for Conformer models.

    Computes a relative positional encoding that covers both positive and
    negative offsets from the current position. The encoding is extended
    dynamically when the input sequence length exceeds the cached length.

    Args:
        d_model (int): Dimensionality of the model embeddings. Default: ``512``.
        max_len (int): Initial maximum sequence length for precomputed encodings. Default: ``5000``.

    Inputs: x
        - **x** (batch, time, d_model): Input tensor whose time dimension
          determines the positional encoding length.

    Returns: pos_emb
        - **pos_emb** (1, 2 * time - 1, d_model): Relative positional encoding
          centered on the current position.

    Examples::

        >>> rel_pe = RelPositionalEncoding(d_model=512)
        >>> x = torch.randn(2, 100, 512)
        >>> enc = rel_pe(x)
        >>> enc.shape
        torch.Size([1, 199, 512])
    """
    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: Tensor) -> None:
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: Tensor) -> Tensor:
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1: self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb
