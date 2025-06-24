import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from ...tokenizers.tokenizer import Tokenizer
from .. import register_criterion
from ..cross_entropy.configuration import CrossEntropyLossConfigs

@register_criterion("cross_entropy", dataclass=CrossEntropyLossConfigs)
class CrossEntropyLoss(nn.Module):
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