from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from asr.encoders import ASREncoder
from asr.modules import Linear, Transpose


class ConvolutionalLSTMEncoder(ASREncoder):
    r"""Convolutional front-end followed by a bidirectional LSTM encoder.

    Applies a convolutional feature extractor (VGG, DeepSpeech2, or
    Conv2dSubSampling) to down-sample the input, then encodes the resulting
    features with a multi-layer bidirectional RNN. Optionally produces CTC
    logits via a fully-connected projection.

    Args:
        input_dim (int): Number of input frequency bins.
        num_classes (int): Number of output token classes (required when
            ``joint_ctc_attention=True``). Default: ``None``.
        hidden_state_dim (int): RNN hidden state size per direction. Default: ``512``.
        dropout_p (float): Dropout probability for RNN layers. Default: ``0.3``.
        num_layers (int): Number of recurrent layers. Default: ``3``.
        bidirectional (bool): Use bidirectional RNN. Default: ``True``.
        rnn_type (str): RNN cell type; one of ``"lstm"``, ``"gru"``, ``"rnn"``.
            Default: ``"lstm"``.
        extractor (str): Convolutional extractor type; one of ``"vgg"``,
            ``"ds2"``, ``"conv2d_subsample"``. Default: ``"vgg"``.
        conv_activation (str): Activation for convolutional layers. Default: ``"hardtanh"``.
        joint_ctc_attention (bool): If ``True``, produce CTC logits. Default: ``False``.

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, input_dim): Input feature tensor.
        - **input_lengths** (batch,): Actual sequence lengths.

    Returns: outputs, encoder_logits, output_lengths
        - **outputs** (batch, time', hidden_state_dim * 2): Encoder hidden states.
        - **encoder_logits** (batch, num_classes, time') or ``None``: CTC log-probabilities.
        - **output_lengths** (batch,): Lengths after convolution.

    Examples::

        >>> encoder = ConvolutionalLSTMEncoder(input_dim=80, extractor="vgg")
        >>> x = torch.randn(2, 100, 80)
        >>> lengths = torch.tensor([100, 80])
        >>> out, logits, out_len = encoder(x, lengths)
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
            inputs: Tensor,
            input_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        encoder_logits = None

        conv_outputs, output_lengths = self.conv(inputs, input_lengths)

        conv_outputs = nn.utils.rnn.pack_padded_sequence(conv_outputs.transpose(0, 1), output_lengths.cpu())
        outputs, hidden_states = self.rnn(conv_outputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs.transpose(0, 1)

        if self.joint_ctc_attention:
            encoder_logits = self.fc(outputs.transpose(1, 2)).log_softmax(dim=2)

        return outputs, encoder_logits, output_lengths
