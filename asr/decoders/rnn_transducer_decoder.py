from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

from asr.decoders import ASRDecoder
from asr.modules import Linear


class RNNTransducerDecoder(ASRDecoder):
    supported_rnns = {
        "lstm": nn.LSTM,
        "gru": nn.GRU,
        "rnn": nn.RNN,
    }

    def __init__(
            self,
            num_classes: int,
            hidden_state_dim: int,
            output_dim: int,
            num_layers: int,
            rnn_type: str = "lstm",
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
            dropout_p: float = 0.2,
    ) -> None:
        super(RNNTransducerDecoder, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.pad_id = (pad_id, )
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(num_classes, hidden_state_dim)
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
        self.out_proj = Linear(hidden_state_dim, output_dim)

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Optional[Tensor] = None,
            hidden_states: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        embedded = self.embedding(inputs)
        if hidden_states is not None:
            outputs, hidden_states = self.rnn(embedded, hidden_states)
        else:
            outputs, hidden_states = self.rnn(embedded)
        
        outputs = self.out_proj(outputs)
        return outputs, hidden_states