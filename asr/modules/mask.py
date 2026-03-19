import torch
from torch import Tensor


def get_attn_pad_mask(inputs: Tensor, input_lengths: Tensor, expand_length: int) -> Tensor:
    r"""Create attention padding mask for padded positions.

    Positions in the mask corresponding to padding are set to ``True``
    (i.e. mask value is 1), which will be used to fill ``-inf`` before
    the softmax in attention.

    Args:
        inputs (Tensor): Input tensor used to derive batch size and shape.
        input_lengths (Tensor): Actual lengths of each sequence in the batch.
        expand_length (int): Length to expand the mask along the query dimension.

    Returns: attn_pad_mask
        - **attn_pad_mask** (batch, expand_length, time): Boolean mask where padded
          positions are ``True``.

    Examples::

        >>> inputs = torch.zeros(2, 10, 80)
        >>> input_lengths = torch.tensor([10, 7])
        >>> mask = get_attn_pad_mask(inputs, input_lengths, expand_length=10)
        >>> mask.shape
        torch.Size([2, 10, 10])
    """

    def get_transformer_non_pad_mask(inputs: Tensor, input_lengths: Tensor) -> Tensor:
        """Padding position is set to 0, either use input_lengths to pad_id."""
        batch_size = inputs.size(0)

        if len(inputs.size()) == 2:
            non_pad_mask = inputs.new_ones(inputs.size())  # B x T
        elif len(inputs.size()) == 3:
            non_pad_mask = inputs.new_ones(inputs.size()[:-1])  # B x T
        else:
            raise ValueError(f"Unsupported input shape {inputs.size()}")

        for i in range(batch_size):
            non_pad_mask[i, input_lengths[i]:] = 0

        return non_pad_mask

    non_pad_mask = get_transformer_non_pad_mask(inputs, input_lengths)
    pad_mask = non_pad_mask.lt(1)
    attn_pad_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return attn_pad_mask


def get_attn_subsequent_mask(seq: Tensor) -> Tensor:
    r"""Create causal (subsequent) attention mask to prevent attending to future positions.

    Generates an upper-triangular matrix of ones, so that position ``i``
    cannot attend to position ``j > i``.

    Args:
        seq (Tensor): Input sequence tensor of shape ``(batch, time)``.

    Returns: subsequent_mask
        - **subsequent_mask** (batch, time, time): Upper-triangular mask where future
          positions are ``1``.

    Examples::

        >>> seq = torch.zeros(2, 10).long()
        >>> mask = get_attn_subsequent_mask(seq)
        >>> mask.shape
        torch.Size([2, 10, 10])
    """
    assert seq.dim() == 2
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)

    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()

    return subsequent_mask
