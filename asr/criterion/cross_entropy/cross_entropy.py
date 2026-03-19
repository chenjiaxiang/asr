import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from ...tokenizers.tokenizer import Tokenizer
from .. import register_criterion
from ..cross_entropy.configuration import CrossEntropyLossConfigs


@register_criterion("cross_entropy", dataclass=CrossEntropyLossConfigs)
class CrossEntropyLoss(nn.Module):
    r"""Cross-entropy loss for seq-to-seq ASR models.

    Wraps ``torch.nn.CrossEntropyLoss`` with padding index masking. Truncates
    or pads logits and targets to the same length before computing the loss.

    Args:
        configs (DictConfig): Configuration object containing
            ``configs.criterion.reduction``.
        tokenizer (Tokenizer): Tokenizer whose ``pad_id`` is used as the
            ``ignore_index`` in the cross-entropy loss.

    Inputs: logits, targets
        - **logits** (batch, time, num_classes): Model output log-probabilities.
        - **targets** (batch, time): Ground-truth token index sequences.

    Returns: loss
        - **loss** (scalar): Scalar cross-entropy loss.

    Examples::

        >>> loss_fn = CrossEntropyLoss(configs, tokenizer)
        >>> logits = torch.randn(2, 10, 100)
        >>> targets = torch.randint(0, 100, (2, 10))
        >>> loss = loss_fn(logits, targets)
    """
    def __init__(
            self,
            configs: DictConfig,
            tokenizer: Tokenizer,
    ) -> None:
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(
            reduction=configs.criterion.reduction,
            ignore_index=tokenizer.pad_id,
        )

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        max_target_length = targets.size(1)
        max_logits_length = logits.size(1)

        if max_logits_length > max_target_length:
            logits = logits[:, :max_target_length, :]
        elif max_target_length > max_logits_length:
            targets = targets[:, :max_logits_length]

        logits = logits.contiguous().view(-1, logits.size(-1))

        return self.cross_entropy_loss(
            logits.contiguous().view(-1, logits.size(-1)),
            targets.contiguous().view(-1),
        )
