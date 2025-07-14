from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from asr.decoders import ASRDecoder
from asr.encoders.transformer_transducer_encoder import TransformerTransducerEncoderLayer
from asr.modules import PositionalEncoding, get_attn_pad_mask, get_attn_subsequent_mask

class TransformerTransducerDecoder(ASRDecoder):
    def __init__(
            self,
            num_classes: int,
            model_dim: int = 512,
            d_ff: int = 2048,
            num_layers: int = 2,
            num_heads: int = 8,
            dropout: float = 0.1,
            max_positional_length: int = 5000,
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
    ) -> None:
        super(TransformerTransducerDecoder, self).__init__()
        self.embedding = nn.Embedding(num_classes, model_dim)
        self.scale = np.sqrt(model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_positional_length)
        self.input_dropout = nn.Dropout(p=dropout)
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.decoder_layers = nn.ModuleList(
            [TransformerTransducerEncoderLayer(model_dim, d_ff, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor
        ) -> Tuple[Tensor, Tensor]:
        batch = input.size(0)

        if len(input.size()) == 1: # validate, evaluation
            inputs = inputs.unsqueeze(1)
            target_lengths = inputs.size(1)

            outputs = self.forward_step(
                decoder_inputs=inputs,
                decoder_input_lengths=input_lengths,
                positional_encoding_length=target_lengths,
            )

        else: # train
            target_lengths = input.size(1)

            outputs = self.forward_step(
                decoder_inputs=inputs,
                decoder_input_lengths=input_lengths,
                positional_encoding_length=target_lengths,
            )

        return outputs, input_lengths

    def forward_step(
            self,
            decoder_inputs: Tensor,
            decoder_input_lengths: Tensor,
            positional_encoding_length: int = 1,
    ) -> Tensor:
        dec_self_attn_pad_mask = get_attn_pad_mask(decoder_inputs, decoder_input_lengths, decoder_inputs.size(1))
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(decoder_inputs)
        self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        embedding_output = self.embedding(decoder_inputs) * self.scale
        positional_encoding_output = self.positional_encoding(positional_encoding_length)
        inputs = embedding_output + positional_encoding_output

        outputs = self.input_dropout(inputs)

        for decoder_layer in self.decoder_layers:
            outputs, _ = decoder_layer(outputs, self_attn_mask)

        return outputs