from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from asr.encoders import ASREncoder
from asr.modules import MultiHeadAttention, PositionalEncoding, PositionwiseFeedForward, get_attn_pad_mask


class TransformerTransducerEncoderLayer(nn.Module):
    r"""Single layer of the Transformer Transducer encoder.

    Applies pre-layer normalization, multi-head self-attention with a
    residual connection, another pre-layer normalization, and a
    position-wise feed-forward network with dropout and a residual connection.

    Args:
        model_dim (int): Model dimensionality. Default: ``512``.
        d_ff (int): Inner dimension of the feed-forward network. Default: ``2048``.
        num_heads (int): Number of attention heads. Default: ``8``.
        dropout (float): Dropout probability. Default: ``0.1``.

    Inputs: inputs, self_attn_mask
        - **inputs** (batch, time, model_dim): Input tensor.
        - **self_attn_mask** (batch, time, time): Optional boolean attention mask.

    Returns: output, attn_distribution
        - **output** (batch, time, model_dim): Layer output.
        - **attn_distribution** (batch, heads, time, time): Attention weights.

    Examples::

        >>> layer = TransformerTransducerEncoderLayer(model_dim=512, num_heads=8)
        >>> x = torch.randn(2, 10, 512)
        >>> out, attn = layer(x)
        >>> out.shape
        torch.Size([2, 10, 512])
    """
    def __init__(
            self,
            model_dim: int = 512,
            d_ff: int = 2048,
            num_heads: int = 8,
            dropout: float = 0.1,
    ) -> None:
        super(TransformerTransducerEncoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(model_dim)
        self.self_attention = MultiHeadAttention(model_dim, num_heads)
        self.encoder_dropout = nn.Dropout(p=dropout)
        self.feed_forward = PositionwiseFeedForward(model_dim, d_ff, dropout)

    def forward(
            self,
            inputs: Tensor,
            self_attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        inputs = self.layer_norm(inputs)
        self_attn_output, attn_distribution = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        self_attn_output += inputs

        self_attn_output = self.layer_norm(self_attn_output)
        ff_output = self.feed_forward(self_attn_output)
        output = self.encoder_dropout(ff_output + self_attn_output)

        return output, attn_distribution


class TransformerTransducerEncoder(ASREncoder):
    r"""Transformer Transducer encoder.

    Encodes input features using sinusoidal positional encoding and a stack
    of transformer encoder layers (with pre-layer normalization). Designed for
    use with the RNN-T / Transformer Transducer objective.

    Args:
        input_size (int): Number of input frequency bins. Default: ``80``.
        model_dim (int): Model dimensionality. Default: ``512``.
        d_ff (int): Inner feed-forward dimension. Default: ``2048``.
        num_layers (int): Number of transformer encoder layers. Default: ``18``.
        num_heads (int): Number of self-attention heads. Default: ``8``.
        dropout (float): Dropout probability. Default: ``0.1``.
        max_positional_lengths (int): Maximum sequence length for positional
            encoding. Default: ``5000``.

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, input_size): Input feature tensor.
        - **input_lengths** (batch,): Actual sequence lengths.

    Returns: outputs, input_lengths
        - **outputs** (batch, time, model_dim): Encoder hidden states.
        - **input_lengths** (batch,): Passed through unchanged.

    Examples::

        >>> encoder = TransformerTransducerEncoder(input_size=80, model_dim=512)
        >>> x = torch.randn(2, 100, 80)
        >>> lengths = torch.tensor([100, 80])
        >>> out, lens = encoder(x, lengths)
        >>> out.shape
        torch.Size([2, 100, 512])
    """
    def __init__(
            self,
            input_size: int = 80,
            model_dim: int = 512,
            d_ff: int = 2048,
            num_layers: int = 18,
            num_heads: int = 8,
            dropout: float = 0.1,
            max_positional_lengths: int = 5000
    ) -> None:
        super(TransformerTransducerEncoder, self).__init__()
        self.input_size = input_size
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_positional_lengths)
        self.input_fc = nn.Linear(input_size, model_dim)
        self.encoder_layers = nn.ModuleList(
            [TransformerTransducerEncoderLayer(model_dim, d_ff, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        seq_len = inputs.size(1)

        self_attn_mask = get_attn_pad_mask(inputs, input_lengths, seq_len)

        inputs = self.input_fc(inputs) + self.positional_encoding(seq_len)
        outputs = self.input_dropout(inputs)

        for encoder_layer in self.encoder_layers:
            outputs, _ = encoder_layer(outputs, self_attn_mask)

        return outputs, input_lengths
