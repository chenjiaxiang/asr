import torch.nn as nn
from torch import Tensor

from asr.modules.conformer_attention_module import MultiHeadedSelfAttentionModule
from asr.modules.conformer_convolution_module import ConformerConvModule
from asr.modules.residual_connection_module import ResidualConnectionModule
from asr.modules.conformer_feed_forward_module import FeedForwardModule


class ConformerBlock(nn.Module):
    r"""Single Conformer encoder block.

    Combines a macaron-style feed-forward module, a multi-headed self-attention
    module with relative positional encoding, a convolution module, and another
    feed-forward module, each wrapped with a residual connection. Ends with
    layer normalization.

    Reference:
        "Conformer: Convolution-augmented Transformer for Speech Recognition"
        - Gulati et al.
        https://arxiv.org/abs/2005.08100

    Args:
        encoder_dim (int): Model dimensionality. Default: ``512``.
        num_attention_heads (int): Number of self-attention heads. Default: ``8``.
        feed_forward_expansion_factor (int): Feed-forward expansion ratio. Default: ``4``.
        conv_expansion_factor (int): Convolution module expansion factor. Default: ``2``.
        feed_forward_dropout_p (float): Dropout for feed-forward modules. Default: ``0.1``.
        attention_dropout_p (float): Dropout for attention module. Default: ``0.1``.
        conv_dropout_p (float): Dropout for convolution module. Default: ``0.1``.
        conv_kernel_size (int): Kernel size for depthwise convolution. Default: ``31``.
        half_step_residual (bool): If ``True``, scale feed-forward residuals by ``0.5``.
            Default: ``True``.

    Inputs: inputs
        - **inputs** (batch, time, encoder_dim): Input tensor.

    Returns: output
        - **output** (batch, time, encoder_dim): Output tensor.

    Examples::

        >>> block = ConformerBlock(encoder_dim=512, num_attention_heads=8)
        >>> x = torch.randn(2, 10, 512)
        >>> out = block(x)
        >>> out.shape
        torch.Size([2, 10, 512])
    """
    def __init__(
            self,
            encoder_dim: int = 512,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
    ) -> None:
        super(ConformerBlock, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1.0

        self.sequential = nn.Sequential(
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            ResidualConnectionModule(
                module=MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                ),
            ),
            ResidualConnectionModule(
                module=ConformerConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                ),
            ),
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            nn.LayerNorm(encoder_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)
