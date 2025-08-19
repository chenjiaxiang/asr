from typing import Optional, Tuple

import torch
import torch.nn as nn

from asr.encoders import ASREncoder
from asr.modules import Linear, Transpose

class ConvolutionalLSTMEncoder(ASREncoder):
    supported_rnns = {
        "lstm": nn.LSTM,
        "gru": nn.GRU,
        "rnn": nn.RNN,
    }
    def __init__(
            self,
            input_dim: int,
            num_classes: int = None,
            hidden_state_dim: int = 512,
            dropout_p: float = 0.3,
            num_layers: int = 3,
            bidirectional: bool = True,
            rnn_type: str = "lstm",
            extractor: str = "vgg",
            conv_activation: str = "hardtanh",
            joint_ctc_attention: bool = False,
    ) -> None:
        super(ConvolutionalLSTMEncoder, self).__init__()
        extractor = self.supported_extractors[extractor.lower()]
        self.conv = extractor(input_dim=input_dim, activation=conv_activation)
        self.conv_output_dim = self.conv.get_output_dim()
        
        self.num_classes = num_classes
        self.joint_ctc_attention = joint_ctc_attention

        self.hidden_state_dim = hidden_state_dim
        self.rnn = self.supported_rnns[rnn_type.lower()](
            input_size=self.conv_output_dim,
            hidden_size=hidden_state_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=bidirectional,
        )

        if self.joint_ctc_attention:
            self.fc = nn.Sequential(
                Transpose(shape=(1, 2)),
                nn.Dropout(dropout_p),
                Linear(hidden_state_dim << 1, num_classes, bias=False),
            )
            
    def forward(
            self,
            inputs: torch.Tensor,
            input_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        encoder_logits = None
        
        conv_outputs, output_lengths = self.conv(inputs, input_lengths)

        conv_outputs = nn.utils.rnn.pack_padded_sequence(conv_outputs.transpose(0, 1), output_lengths.cpu())
        outputs, hidden_states = self.rnn(conv_outputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs.transpose(0, 1)

        if self.joint_ctc_attention:
            encoder_logits = self.fc(outputs.transpose(1, 2)).log_softmax(dim=2)
        
        return outputs, encoder_logits, output_lengths