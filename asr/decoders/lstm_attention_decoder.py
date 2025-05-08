import random
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from asr.decoders import ASRDecoder
from asr.modules import (
    AdditiveAttention,
    DotProductAttention,
    Linear,
    LocationAwareAttention,
    MultiHeadAttention,
    View,
)


class LSTMAttentionDecoder(ASRDecoder):
    supported_rnns = {
        "lstm": nn.LSTM,
        "gru": nn.GRU,
        "rnn": nn.RNN,
    }

    def __init__(
            self,
            num_classes: int,
            max_lengths: int = 150,
            hidden_state_dim: int = 1024,
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
            attn_mechanism: str = "multi-head",
            num_heads: int = 4,
            num_layers: int = 2,
            rnn_type: str = "lstm",
            dropout_p: float = 0.3,
    ) -> None:
        super(LSTMAttentionDecoder, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_length = max_lengths
        self.eos_id = eos_id
        self.sos_id = pad_id   
        self.pad_id = pad_id
        self.attn_mechanism = attn_mechanism.lower()
        self.emebedding = nn.Embedding(num_classes, hidden_state_dim)
        self.input_dropout = nn.Dropout(dropout_p)
        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.rnn = rnn_cell(
            input_size=hidden_state_dim,
            hidden_size=hidden_state_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=False,
        )

        if self.attn_mechanism == "loc":
            self.attention = LocationAwareAttention(hidden_state_dim, attn_dim=hidden_state_dim, smoothing=False)
        elif self.attn_mechanism == "multi-head":
            self.attention = MultiHeadAttention(hidden_state_dim, num_heads=num_heads)
        elif self.attn_mechanism == "additive":
            self.attention = AdditiveAttention(hidden_state_dim)
        elif self.attn_mechanism == "dot":
            self.attention = DotProductAttention(dim=hidden_state_dim)
        elif self.attn_mechanism == "scaled-dot":
            self.attention = DotProductAttention(dim=hidden_state_dim, scale=True)
        else:
            raise ValueError("Unsupported attention: %s".format(attn_mechanism))

        self.fc = nn.Sequential(
            Linear(hidden_state_dim << 1, hidden_state_dim),
            nn.Tanh(),
            View(shape=(-1, self.hidden_state_dim), contiguous=True),
            Linear(hidden_state_dim, num_classes),
        )

    def forward_step(
            self,
            input_var: Tensor,
            hidden_states: Optional[Tensor],
            encoder_outputs: Tensor,
            attn: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, output_lengths = input_var.size(0), input_var.size(1)
        
        embedded = self.emebedding(input_var)
        embedded = self.input_dropout(embedded)

        if self.training:
            self.rnn.flatten_parameters()

        outputs, hidden_states = self.rnn(embedded, hidden_states)

        if self.attn_mechanism == "loc":
            context, attn = self.attention(outputs, encoder_outputs, attn)
        else:
            context, attn = self.attention(outputs, encoder_outputs, encoder_outputs)
        
        outputs = torch.cat((outputs, context), dim=2)

        step_outputs = self.fc(outputs.view(-1, self.hidden_state_dim << 1)).log_softmax(dim=-1)
        step_outputs = step_outputs.view(batch_size, output_lengths, -1).squeeze(1)

        return step_outputs, hidden_states, attn

    def forward(
            self,
            encoder_outputs: Tensor,
            targets: Optional[Tensor] = None,
            encoder_output_lengths: Optional[Tensor] = None,
            teacher_forcing_ratio: float = 1.0,
    ) -> Tensor:
        logits = list()
        hidden_states, attn = None, None

        targets, batch_size, max_lengths = self.validate_args(targets, encoder_outputs, teacher_forcing_ratio)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            targets = targets[targets != self.eos_id].view(batch_size, -1)

            if self.attn_mechanism == "loc" or self.attn_mechanism == "additive":
                for di in range(targets.size[1]):
                    input_var = targets[:, di].unsqueeze(1)
                    step_outputs, hidden_states, attn = self.forward_step(
                        input_var=input_var,
                        hidden_states=hidden_states,
                        encoder_outputs=encoder_outputs,
                        attn=attn,
                    )
                    logits.append(step_outputs)

            else:
                step_outputs, hidden_states, attn = self.forward_step(
                    input_var=targets,
                    hidden_states=hidden_states,
                    encoder_outputs=encoder_outputs,
                    attn=attn,
                )

                for di in range(step_outputs.size(1)):
                    step_outputs = step_outputs[:, di, :]
                    logits.append(step_outputs)

        else:
            input_var = targets[:, 0].unsqueeze(1)

            for di in range(max_lengths):
                step_outputs, hidden_states, attn = self.forward_step(
                    input_var=input_var,
                    hidden_states=hidden_states,
                    encoder_outputs=encoder_outputs,
                    attn=attn,
                )
                logits.append(step_outputs)
                input_var = logits[-1].topk(1)[1]

        logits = torch.stack(logits, dim=-1)

        return logits

    def validate_args(
            self,
            targets: Optional[Any] = None,
            encoder_outputs: Tensor = None,
            teacher_forcing_ratio: float = 1.0,
    ) -> Tuple[Tensor, int, int]:
        assert encoder_outputs is not None
        batch_size = encoder_outputs.size(0)

        if targets is None:
            targets = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            max_length = self.max_length

            if torch.cuda.is_available():
                targets = targets.cuda()

            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no targets is provided.")

        else:
            max_length = targets.size(1) - 1

        return targets, batch_size, max_length
