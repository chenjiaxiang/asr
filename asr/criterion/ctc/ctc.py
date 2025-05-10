import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from ...tokenizers.tokenizer import Tokenizer
from .. import register_criterion
from ..ctc.configuration import CTCLossConfigs


@register_criterion("ctc", dataclass=CTCLossConfigs)
class CTCLoss(nn.Module):
    def __init__(
            self,
            configs: DictConfig,
            tokenizer: Tokenizer,
    ) -> None:
        super(CTCLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(
            blank=tokenizer.blank_id,
            reduction=configs.criterion.reduction,
            zero_infinity=configs.criterion.zero_infinity,
        )

    def forward(
            self,
            log_probs: Tensor,
            input_lengths: Tensor,
            targets: Tensor,
            target_lengths: Tensor,
    ) -> Tensor:
        return self.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
        )