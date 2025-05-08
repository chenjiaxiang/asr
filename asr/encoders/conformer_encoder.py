from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from asr.encoders.asr_encoder import ASREncoder
from asr.modules import ConformerBlock, Conv2dSubSampling, Linear, Transpose


class ConformerEncoder(ASREncoder):
    def __init__(
            self,
            num_classes: int,
            input_dim: int = 80,
            encoder_dim: int = 512,
            num_layers: int = 17,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
            joint_ctc_attention: bool = True,
    ) -> None:
        super(ConformerEncoder, self).__init__()
        self.joint_ctc_attention = joint_ctc_attention
        self.conv_subsample = Conv2dSubSampling(input_dim, in_channels=1, out_channels=encoder_dim)
        self.input_projection = nn.Sequential(
            Linear(self.conv_subsample.get_output_dim(), encoder_dim),
            nn.Dropout(p=input_dropout_p),
        )
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    encoder_dim=encoder_dim,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                    conv_kernel_size=conv_kernel_size,
                    half_step_residual=half_step_residual,
                )
                for _ in range(num_layers)
            ]
        )
        if self.joint_ctc_attention:
            self.fc = nn.Sequential(
                Transpose(shape=(1, 2)),
                nn.Dropout(feed_forward_dropout_p),
                Linear(encoder_dim, num_classes, bias=False),
            )

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
        ) -> Tuple[Tensor, Tensor, Tensor]:
        encoder_logits = None

        outputs, output_lengths = self.conv_subsample(inputs, input_lengths)
        outputs = self.input_projection(outputs)

        for layer in self.layers:
            outputs = layer(outputs)

        if self.joint_ctc_attention:
            encoder_logits = self.fc(outputs.transpose(1, 2)).log_softmax(dim=2)

        return outputs, encoder_logits, output_lengths