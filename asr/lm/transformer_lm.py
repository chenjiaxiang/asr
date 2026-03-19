from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from asr.lm.asr_lm import ASRLanguageModelBase
from asr.modules import (
    Linear,
    MultiHeadAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    TransformerEmbedding,
    get_attn_pad_mask,
    get_attn_subsequent_mask,
)


class TransformerForLanguageModelLayer(nn.Module):
    r"""Single transformer layer for the language model.

    Applies pre-layer normalization, multi-head self-attention with a
    residual connection, then pre-layer normalization and a feed-forward
    network with a residual connection.

    Args:
        d_model (int): Model dimensionality. Default: ``768``.
        num_attention_heads (int): Number of attention heads. Default: ``8``.
        d_ff (int): Feed-forward inner dimensionality. Default: ``2048``.
        dropout_p (float): Dropout probability. Default: ``0.3``.

    Inputs: inputs, mask
        - **inputs** (batch, time, d_model): Input tensor.
        - **mask** (batch, time, time): Optional boolean attention mask.

    Returns: outputs
        - **outputs** (batch, time, d_model): Layer output tensor.

    Examples::

        >>> layer = TransformerForLanguageModelLayer(d_model=768)
        >>> x = torch.randn(2, 10, 768)
        >>> out = layer(x)
        >>> out.shape
        torch.Size([2, 10, 768])
    """
    def __init__(
            self,
            d_model: int = 768,
            num_attention_heads: int = 8,
            d_ff: int = 2048,
            dropout_p: float = 0.3,
    ) -> None:
        super(TransformerForLanguageModelLayer, self).__init__()
        self.attention_prenorm = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_attention_heads)
        self.feed_forward_prenorm = nn.LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout_p=dropout_p)

    def forward(
            self,
            inputs: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
        residual = inputs
        inputs = self.attention_prenorm(inputs)
        outputs, _ = self.attention(inputs, inputs, inputs, mask)
        outputs += residual

        residual = outputs
        outputs = self.feed_forward_prenorm(outputs)
        outputs = self.feed_forward(outputs)
        outputs += residual

        return outputs


class TransformerForLanguageModel(ASRLanguageModelBase):
    r"""Transformer-based language model for ASR rescoring or shallow fusion.

    Encodes token sequences with sinusoidal positional encoding and a stack
    of causal (masked) transformer layers. Produces log-probability
    distributions over the vocabulary at each position.

    Args:
        num_classes (int): Vocabulary size.
        max_length (int): Maximum sequence length. Default: ``128``.
        d_model (int): Model dimensionality. Default: ``768``.
        num_attention_heads (int): Number of attention heads. Default: ``8``.
        d_ff (int): Feed-forward inner dimensionality. Default: ``1536``.
        pad_id (int): Padding token index. Default: ``0``.
        sos_id (int): Start-of-sequence token index. Default: ``1``.
        eos_id (int): End-of-sequence token index. Default: ``2``.
        num_layers (int): Number of transformer layers. Default: ``2``.
        dropout_p (float): Dropout probability. Default: ``0.3``.

    Inputs: inputs, input_lengths
        - **inputs** (batch, time): Token index tensor.
        - **input_lengths** (batch,): Sequence lengths.

    Returns: logits
        - **logits** (batch, time, num_classes): Log-probability distributions.

    Examples::

        >>> lm = TransformerForLanguageModel(num_classes=100)
        >>> x = torch.randint(0, 100, (2, 20))
        >>> lengths = torch.tensor([20, 15])
        >>> out = lm(x, lengths)
        >>> out.shape
        torch.Size([2, 20, 100])
    """
    def __init__(
        self,
        num_classes: int,
        max_length: int = 128,
        d_model: int = 768,
        num_attention_heads: int = 8,
        d_ff: int = 1536,
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
        num_layers: int = 2,
        dropout_p: float = 0.3,
    ) -> None:
        super(TransformerForLanguageModel, self).__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.max_length = max_length
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.pad_id = pad_id
        self.embedding = TransformerEmbedding(num_classes, pad_id, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.layers = nn.ModuleList(
            [
                TransformerForLanguageModelLayer(
                    d_model=d_model,
                    num_attention_heads=num_attention_heads,
                    d_ff=d_ff,
                    dropout_p=dropout_p,
                )
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            Linear(d_model, d_model, bias=False),
            nn.Tanh(),
            Linear(d_model, num_classes, bias=False),
        )

    def forward_step(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        pad_mask = get_attn_pad_mask(inputs, input_lengths, inputs.size(1))
        subsequent_mask = get_attn_subsequent_mask(inputs)
        mask = torch.gt((pad_mask + subsequent_mask), 0)

        outputs = self.embedding(inputs) + self.positional_encoding(inputs.size(1))
        outputs = self.input_dropout(outputs)

        for layer in self.layers:
            outputs = layer(inputs=outputs, mask=mask)

        step_outputs = self.fc(outputs).log_softmax(dim=-1)

        return step_outputs

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        logits = list()

        step_outputs = self.forward_step(inputs, input_lengths)

        for di in range(step_outputs.size(1)):
            step_output = step_outputs[:, di, :]
            logits.append(step_output)

        return torch.stack(logits, dim=1)
