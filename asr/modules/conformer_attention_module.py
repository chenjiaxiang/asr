from typing import Optional

import torch.nn as nn
from torch import Tensor

from asr.modules.positional_encoding import RelPositionalEncoding
from asr.modules.relative_multi_head_attention import RelativeMultiHeadAttention

class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            dropout_p: float = 0.1,
    ) -> None:
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.postional_encoding = RelPositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout= nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size = inputs.size(0)
        pos_embedding = self.postional_encoding(inputs)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)

        return self.dropout(outputs)