from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from asr.modules.depthwise_conv2d import DepthWiseConv2d


class MaskConv2d(nn.Module):
    r"""2D convolutional sequence with masking for variable-length inputs.

    Wraps an ``nn.Sequential`` of 2D convolutional (and optional pooling)
    layers and ensures that padded positions are zeroed out after each layer.
    Sequence lengths are updated at each layer to reflect the output length
    after striding.

    Args:
        sequential (nn.Sequential): A sequential container of ``nn.Conv2d``,
            ``DepthWiseConv2d``, and/or ``nn.MaxPool2d`` modules (interleaved
            with activation functions).

    Inputs: inputs, seq_lengths
        - **inputs** (batch, channels, freq, time): Input feature tensor.
        - **seq_lengths** (batch,): Actual time lengths of each sequence.

    Returns: output, seq_lengths
        - **output** (batch, channels, freq', time'): Masked output tensor.
        - **seq_lengths** (batch,): Updated time lengths after all layers.

    Examples::

        >>> conv = MaskConv2d(nn.Sequential(
        ...     nn.Conv2d(1, 32, kernel_size=3, stride=2),
        ...     nn.ReLU(),
        ... ))
        >>> x = torch.randn(2, 1, 80, 100)
        >>> lengths = torch.tensor([100, 80])
        >>> out, out_len = conv(x, lengths)
    """
    def __init__(self, sequential: nn.Sequential) -> None:
        super(MaskConv2d, self).__init__()
        self.sequential = sequential

    def forward(self, inputs: Tensor, seq_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        output = None

        for module in self.sequential:
            output = module(inputs)
            mask = torch.BoolTensor(output.size()).fill_(0)

            if output.is_cuda:
                mask = mask.cuda()

            seq_lengths = self._get_sequence_lengths(module, seq_lengths)

            for idx, length in enumerate(seq_lengths):
                lengths = length.item()

                if (mask[idx].size(2) - lengths) > 0:
                    mask[idx].narrow(dim=2, start=length, length=mask[idx].size(2) - length).fill_(1)

            output = output.masked_fill(mask, 0)
            inputs = output

        return output, seq_lengths

    def _get_sequence_lengths(self, module: nn.Module, seq_lengths: Tensor) -> Tensor:
        if isinstance(module, nn.Conv2d):
            numerator = seq_lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1
            seq_lengths = numerator.float() / float(module.stride[1])
            seq_lengths = seq_lengths.int() + 1
        elif isinstance(module, DepthWiseConv2d):
            numerator = (
                seq_lengths
                + 2 * module.conv.padding[1]
                - module.conv.dilation[1] * (module.conv.kernel_size[1] - 1)
                - 1
            )
            seq_lengths = numerator.float() / float(module.conv.stride[1])
            seq_lengths = seq_lengths.int() + 1
        elif isinstance(module, nn.MaxPool2d):
            seq_lengths >>= 1

        return seq_lengths.int()
