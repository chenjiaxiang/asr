from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from asr.encoders import ASREncoder
from asr.modules import Linear, Transpose


class LSTMEncoder(ASREncoder):
    r"""Bidirectional LSTM encoder for ASR.

    Encodes variable-length input sequences using a multi-layer bidirectional
    RNN (LSTM, GRU, or vanilla RNN). Optionally computes CTC logits from the
    encoder outputs when ``joint_ctc_attention=True``.

    Args:
        input_dim (int): Dimensionality of each input frame.
        num_classes (int): Number of output token classes (required when
            ``joint_ctc_attention=True``). Default: ``None``.
        hidden_state_dim (int): Number of features in the hidden state of each
            RNN direction. Default: ``512``.
        dropout_p (float): Dropout probability between RNN layers. Default: ``0.3``.
        num_layers (int): Number of recurrent layers. Default: ``3``.
        bidirectional (bool): If ``True``, use a bidirectional RNN. Default: ``True``.
        rnn_type (str): Type of RNN cell; one of ``"lstm"``, ``"gru"``, ``"rnn"``.
            Default: ``"lstm"``.
        joint_ctc_attention (bool): If ``True``, produce CTC logits. Default: ``False``.

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, input_dim): Padded input features.
        - **input_lengths** (batch,): Actual length of each sequence.

    Returns: outputs, encoder_logits, input_lengths
        - **outputs** (batch, time, hidden_state_dim * 2): Encoder hidden states.
        - **encoder_logits** (batch, num_classes, time) or ``None``: CTC log-probabilities.
        - **input_lengths** (batch,): Passed through unchanged.

    Examples::

        >>> encoder = LSTMEncoder(input_dim=80, hidden_state_dim=512)
        >>> x = torch.randn(2, 100, 80)
        >>> lengths = torch.tensor([100, 80])
        >>> out, logits, lens = encoder(x, lengths)
        >>> out.shape
        torch.Size([2, 100, 1024])
    """
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
            joint_ctc_attention: bool = False,
    ) -> None:
        super(LSTMEncoder, self).__init__()

        self.num_classes = num_classes
        self.joint_ctc_attention = joint_ctc_attention

        self.hidden_state_dim = hidden_state_dim
        self.rnn = self.supported_rnns[rnn_type.lower()](
            input_size=input_dim,
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
            inputs: Tensor,
            input_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        encoder_logits = None

        conv_outputs = nn.utils.rnn.pack_padded_sequence(inputs.transpose(0, 1), input_lengths.cpu())
        outputs, hidden_states = self.rnn(conv_outputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs.transpose(0, 1)

        if self.joint_ctc_attention:
            encoder_logits = self.fc(outputs.transpose(1, 2)).log_softmax(dim=2)

        return outputs, encoder_logits, input_lengths
