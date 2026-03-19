import math
import torch.nn as nn
from torch import Tensor


class TransformerEmbedding(nn.Module):
    r"""Token embedding for transformer models with ``sqrt(d_model)`` scaling.

    Wraps ``torch.nn.Embedding`` and scales the output by ``sqrt(d_model)``
    as described in "Attention Is All You Need" (Vaswani et al., 2017).

    Args:
        num_embeddings (int): Size of the vocabulary (number of token classes).
        pad_id (int): Index of the padding token; its embedding is set to zero.
        d_model (int): Dimensionality of the embeddings. Default: ``512``.

    Inputs: inputs
        - **inputs** (batch, time): LongTensor of token indices.

    Returns: output
        - **output** (batch, time, d_model): Scaled embedding tensor.

    Examples::

        >>> emb = TransformerEmbedding(num_embeddings=1000, pad_id=0, d_model=512)
        >>> x = torch.randint(0, 1000, (2, 10))
        >>> out = emb(x)
        >>> out.shape
        torch.Size([2, 10, 512])
    """
    def __init__(self, num_embeddings: int, pad_id: int, d_model: int = 512) -> None:
        super(TransformerEmbedding, self).__init__()
        self.sqrt_dim = math.sqrt(d_model)
        self.embedding = nn.Embedding(num_embeddings, d_model, padding_idx=pad_id)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.embedding(inputs) * self.sqrt_dim
