from typing import Tuple

import torch.nn as nn
from torch import Tensor

from asr.modules import Conv2dSubSampling, DeepSpeed2Extractor, Swish, VGGExtractor


class ASREncoder(nn.Module):
    r"""Abstract base class for ASR encoders.

    Provides common infrastructure shared by all encoder implementations,
    including a registry of supported activation functions and feature
    extractors, a parameter counter, and a dropout updater.

    Subclasses must implement :meth:`forward`.

    Args:
        (none — subclasses accept their own constructor arguments)

    Examples::

        >>> class MyEncoder(ASREncoder):
        ...     def forward(self, inputs, input_lengths):
        ...         return inputs, input_lengths
    """
    supported_activations = {
        "hardtanh": nn.Hardtanh(0, 20, inplace=True),
        "relu": nn.ReLU(inplace=True),
        "elu": nn.ELU(inplace=True),
        "leaky_elu": nn.LeakyReLU(inplace=True),
        "gelu": nn.GELU(),
        "Swish": Swish(),
    }
    supported_extractors = {
        "ds2": DeepSpeed2Extractor,
        "vgg": VGGExtractor,
        "conv2d_subsample": Conv2dSubSampling,
    }

    def __init__(self) -> None:
        super(ASREncoder, self).__init__()

    def count_parameters(self) -> int:
        return sum([p.numel() for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, ...]:
        raise NotImplementedError
