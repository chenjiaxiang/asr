import math
import torch.nn as nn
from torch import Tensor

class TransformerEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, pad_id: int, d_model: int = 512) -> None:
        super(TransformerEmbedding, self).__init__()
        self.sqrt_dim = math.sqrt(d_model)
        self.embedding = nn.Embedding(num_embeddings, d_model, padding_idx=pad_id)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.embedding(inputs) * self.sqrt_dim