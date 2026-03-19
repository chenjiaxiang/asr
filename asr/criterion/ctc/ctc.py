import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from ...tokenizers.tokenizer import Tokenizer
from .. import register_criterion
from ..ctc.configuration import CTCLossConfigs


@register_criterion("ctc", dataclass=CTCLossConfigs)
class CTCLoss(nn.Module):
    r"""Connectionist Temporal Classification (CTC) loss.

    Wraps ``torch.nn.CTCLoss`` with configuration from a Hydra/OmegaConf
    config and a tokenizer that provides the blank token index.

    Args:
        configs (DictConfig): Configuration object containing
            ``configs.criterion.reduction`` and
            ``configs.criterion.zero_infinity``.
        tokenizer (Tokenizer): Tokenizer whose ``blank_id`` attribute is used
            as the CTC blank token index.

    Inputs: log_probs, input_lengths, targets, target_lengths
        - **log_probs** (time, batch, num_classes): Log-probabilities from the
          model (output of ``log_softmax``).
        - **input_lengths** (batch,): Lengths of the log-prob sequences.
        - **targets** (batch, target_len): Ground-truth token sequences.
        - **target_lengths** (batch,): Lengths of the target sequences.

    Returns: loss
        - **loss** (scalar): Scalar CTC loss.

    Examples::

        >>> loss_fn = CTCLoss(configs, tokenizer)
        >>> log_probs = torch.randn(50, 2, 100).log_softmax(dim=-1)
        >>> targets = torch.randint(1, 100, (2, 10))
        >>> loss = loss_fn(log_probs, torch.tensor([50, 50]), targets, torch.tensor([10, 8]))
    """
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
