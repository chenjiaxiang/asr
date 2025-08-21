from typing import Optional, Tuple

import torch
import torch.nn as nn

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
            inputs: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def forward_step(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        pad_mask = get_attn_pad_mask(inputs, input_lengths, inputs.size(1))
        subsequent_mask = get_attn_subsequent_mask(inputs)
        mask = torch.gt((pad_mask + subsequent_mask), 0)

        outputs = self.embedding(inputs) + self.positional_encoding(inputs.size(1))
        outputs = self.input_dropout(outputs)

        for layer in self.layers:
            outputs = layer(inputs=outputs, mask=mask)

        step_outputs = self.fc(outputs).log_softmax(dim=-1)

        return step_outputs

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        logits = list()

        step_outputs = self.forward_step(inputs, input_lengths)

        for di in range(step_outputs.size(1)):
            step_output = step_outputs[:, di, :]
            logits.append(step_output)

        return torch.stack(logits, dim=1)