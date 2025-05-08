import torch.nn as nn
from torch import Tensor

from asr.modules import Conv2dSubSampling, DeepSpeed2Extractor, Swish, VGGExtractor

class ASREncoder(nn.Module):
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

    def __init__(self):
        super(ASREncoder, self).__init__()

    def count_parameters(self) -> int:
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p
            
    def forward(self, inputs: Tensor, input_lengths: Tensor):
        raise NotImplementedError
