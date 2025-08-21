import random
from typing import Optional, Tuple

import torch
import torch.nn as nn

from asr.lm.asr_lm import ASRLanguageModelBase
from asr.modules import Linear, View


class LSTMForLanguageModel(ASRLanguageModelBase):
    supported_rnns = {
        "lstm": nn.LSTM,
        "gru": nn.GRU,
        "rnn": nn.RNN,
    }

    def __init__(
            self,
            num_classes: int,
            max_length: int = 128,
            hidden_state_dim: int = 768,
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
            num_layers: int = 2,
            rnn_type: str = "lstm",
            dropout_p: float = 0.3,
    ) -> None:
        super(LSTMForLanguageModel, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.max_length = max_length
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.pad_id = pad_id
        self.embedding = nn.Embedding(num_classes, hidden_state_dim)
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

        self.fc = nn.Sequential(
            Linear(hidden_state_dim, hidden_state_dim),
            nn.Tanh(),
            View(shape=(-1, self.hidden_state_dim), contiguous=True),
            Linear(hidden_state_dim, num_classes),
        )

    def forward_step(
            self,
            input_var: torch.Tensor,
            hidden_states: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, output_lengths = input_var.size(0), input_var.size(1)

        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        if self.training:
            self.rnn.flatten_parameters()

        outputs, hidden_states = self.rnn(embedded, hidden_states)

        step_outputs = self.fc(outputs.reshape(-1, self.hidden_state_dim)).log_softmax(dim=-1)
        step_outputs = step_outputs.view(batch_size, output_lengths, -1).squeeze(1)

        return step_outputs, hidden_states

    def forward(
            self,
            inputs: torch.Tensor,
            teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        batch_size = inputs.size(0)
        logits, hidden_states = list(), None
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            inputs = inputs[inputs != self.eos_id].view(batch_size, -1)
            step_outputs, hidden_states = self.forward_step(input_var=inputs, hidden_states=hidden_states)

            for di in range(step_outputs.size(1)):
                step_output = step_outputs[:, di, :]
                logits.append(step_output)
            
        else:
            input_var = input[:, 0].unsqueeze(1)
            for di in range(self.max_length):
                step_output, hidden = self.forward_step(input_var=input_var, hidden_states=hidden_states)

                step_output = step_output.squeeze(1)
                logits.append(step_output)
                input_var = logits[-1].topk(1)[1]
        
        return torch.stack(logits, dim=1)




