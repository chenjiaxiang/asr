import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from asr.decoders import ASRDecoder
from asr.modules import (
    Linear,
    MultiHeadAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    TransformerEmbedding,
    get_attn_pad_mask,
    get_attn_subsequent_mask,
)


class TransformerDecoderLayer(nn.Module):
    r"""Single layer of the Transformer decoder.

    Applies pre-layer normalization, masked self-attention, cross-attention
    over encoder outputs, and a position-wise feed-forward network, each
    with a residual connection.

    Args:
        d_model (int): Model dimensionality. Default: ``512``.
        num_heads (int): Number of attention heads. Default: ``8``.
        d_ff (int): Feed-forward inner dimensionality. Default: ``2048``.
        dropout_p (float): Dropout probability. Default: ``0.3``.

    Inputs: inputs, encoder_outputs, self_attn_mask, encoder_attn_mask
        - **inputs** (batch, target_len, d_model): Decoder input tensor.
        - **encoder_outputs** (batch, src_len, d_model): Encoder hidden states.
        - **self_attn_mask** (batch, target_len, target_len): Causal + padding mask.
        - **encoder_attn_mask** (batch, target_len, src_len): Encoder padding mask.

    Returns: outputs, self_attn, encoder_attn
        - **outputs** (batch, target_len, d_model): Layer output.
        - **self_attn** (batch, heads, target_len, target_len): Self-attention weights.
        - **encoder_attn** (batch, heads, target_len, src_len): Cross-attention weights.

    Examples::

        >>> layer = TransformerDecoderLayer(d_model=512, num_heads=8)
        >>> tgt = torch.randn(2, 10, 512)
        >>> mem = torch.randn(2, 20, 512)
        >>> out, sa, ca = layer(tgt, mem)
        >>> out.shape
        torch.Size([2, 10, 512])
    """
    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 8,
            d_ff: int = 2048,
            dropout_p: float = 0.3,
    ) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention_prenorm = nn.LayerNorm(d_model)
        self.decoder_attention_prenorm = nn.LayerNorm(d_model)
        self.feed_forward_prenorm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.decoder_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_p)

    def forward(
            self,
            inputs: Tensor,
            encoder_outputs: Tensor,
            self_attn_mask: Optional[Tensor] = None,
            encoder_attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        residual = inputs
        inputs = self.self_attention_prenorm(inputs)
        outputs, self_attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        outputs += residual

        residual = outputs
        outputs = self.decoder_attention_prenorm(outputs)
        outputs, encoder_attn = self.decoder_attention(outputs, encoder_outputs, encoder_outputs, encoder_attn_mask)
        outputs += residual

        residual = outputs
        outputs = self.feed_forward_prenorm(outputs)
        outputs = self.feed_forward(outputs)
        outputs += residual

        return outputs, self_attn, encoder_attn


class TransformerDecoder(ASRDecoder):
    r"""Transformer decoder for seq-to-seq ASR.

    Decodes encoder outputs auto-regressively using a stack of transformer
    decoder layers. Supports teacher forcing during training and greedy
    decoding at inference time.

    Args:
        num_classes (int): Number of output token classes.
        d_model (int): Model dimensionality. Default: ``512``.
        d_ff (int): Feed-forward inner dimensionality. Default: ``512``.
        num_layers (int): Number of decoder layers. Default: ``6``.
        num_heads (int): Number of attention heads. Default: ``8``.
        dropout_p (float): Dropout probability. Default: ``0.3``.
        pad_id (int): Padding token index. Default: ``0``.
        sos_id (int): Start-of-sequence token index. Default: ``1``.
        eos_id (int): End-of-sequence token index. Default: ``2``.
        max_length (int): Maximum decoding length at inference. Default: ``128``.

    Inputs: encoder_outputs, targets, encoder_output_lengths, teacher_forcing_ratio
        - **encoder_outputs** (batch, src_len, d_model): Encoder hidden states.
        - **targets** (batch, target_len): Target token indices (optional).
        - **encoder_output_lengths** (batch,): Encoder output lengths.
        - **teacher_forcing_ratio** (float): Teacher forcing probability. Default: ``1.0``.

    Returns: logits
        - **logits** (batch, target_len, num_classes): Log-probability distributions.

    Examples::

        >>> decoder = TransformerDecoder(num_classes=100, d_model=512)
        >>> enc_out = torch.randn(2, 20, 512)
        >>> targets = torch.randint(0, 100, (2, 10))
        >>> logits = decoder(enc_out, targets)
        >>> logits.shape
        torch.Size([2, 9, 100])
    """
    def __init__(
            self,
            num_classes: int,
            d_model: int = 512,
            d_ff: int = 512,
            num_layers: int = 6,
            num_heads: int = 8,
            dropout_p: float = 0.3,
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
            max_length: int = 128,
    ) -> None:
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id

        self.embedding = TransformerEmbedding(num_classes, pad_id, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
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

    def forward_step(
            self,
            decoder_inputs: Tensor,
            decoder_input_lengths: Tensor,
            encoder_outputs: Tensor,
            encoder_output_lengths: Tensor,
            positional_encoding_length: int,
    ) -> Tensor:
        dec_self_attn_pad_mask = get_attn_pad_mask(decoder_inputs, decoder_input_lengths, decoder_inputs.size(1))
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(decoder_inputs)
        self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        encoder_attn_mask = get_attn_pad_mask(encoder_outputs, encoder_output_lengths, decoder_inputs.size(1))

        outputs = self.embedding(decoder_inputs) + self.positional_encoding(positional_encoding_length)
        outputs = self.input_dropout(outputs)

        for layer in self.layers:
            outputs, self_attn, memory_attn = layer(
                inputs=outputs,
                encoder_outputs=encoder_outputs,
                self_attn_mask=self_attn_mask,
                encoder_attn_mask=encoder_attn_mask,
            )
        return outputs

    def forward(
            self,
            encoder_outputs: Tensor,
            targets: Optional[torch.LongTensor] = None,
            encoder_output_lengths: Tensor = None,
            teacher_forcing_ratio: float = 1.0,
    ) -> Tensor:
        logits = list()
        batch_size = encoder_outputs.size(0)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if targets is not None and use_teacher_forcing:
            targets = targets[targets != self.eos_id].view(batch_size, -1)
            targets_length = targets.size(1)

            step_outputs = self.forward_step(
                decoder_inputs=targets,
                decoder_input_lengths=targets_length,
                encoder_outputs=encoder_outputs,
                encoder_output_lengths=encoder_output_lengths,
                positional_encoding_length=targets_length,
            )
            self_outputs = self.fc(step_outputs).log_softmax(dim=-1)

            for di in range(step_outputs.size(1)):
                step_outputs = step_outputs[:, di, :]
                logits.append(step_outputs)
        else:
            input_var = encoder_outputs.new_zeros(batch_size, self.max_length).long()
            input_var = input_var.fill_(self.pad_id)
            input_var[:, 0] = self.sos_id

            for di in range(1, self.max_length):
                input_lengths = torch.IntTensor(batch_size).fill_(di)

                outputs = self.forward_step(
                    decoder_inputs=input_var[:, :di],
                    decoder_input_lengths=input_lengths,
                    encoder_outputs=encoder_outputs,
                    encoder_output_lengths=encoder_output_lengths,
                    positional_encoding_length=di,
                )
                step_outputs = self.fc(outputs).log_softmax(dim=-1)

                logits.append(step_outputs[:, -1, :])
                input_var[:, di] = logits[-1].topk(1)[1].squeeze()

        return torch.stack(logits, dim=1)
