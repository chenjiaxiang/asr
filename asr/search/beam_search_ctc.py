from typing import Optional

import torch.nn as nn
from torch import Tensor

from asr.utils import CTCDECODE_IMPORT_ERROR


class BeamSearchCTC(nn.Module):
    r"""CTC beam search decoder using ``ctcdecode``.

    Wraps the ``CTCBeamDecoder`` from the ``ctcdecode`` package to perform
    beam search decoding of CTC log-probabilities.

    Args:
        labels (list): List of label strings corresponding to each output class.
        lm_path (str): Path to an n-gram language model file for shallow fusion.
            Default: ``None``.
        alpha (int): Language model weight. Default: ``0``.
        beta (int): Word insertion penalty. Default: ``0``.
        cutoff_top_n (int): Number of top characters to keep per step. Default: ``40``.
        cutoff_prob (float): Minimum probability threshold. Default: ``1.0``.
        beam_size (int): Number of beams to maintain. Default: ``3``.
        num_processes (int): Number of parallel decoding processes. Default: ``4``.
        blank_id (int): Index of the CTC blank token. Default: ``0``.

    Inputs: logits, sizes
        - **logits** (batch, time, num_classes): CTC log-probabilities.
        - **sizes** (batch,): Optional sequence lengths.

    Returns: outputs
        - **outputs** (batch, beam_size, time): Top beam sequences.

    Examples::

        >>> decoder = BeamSearchCTC(labels=list("abcde "), beam_size=3)
        >>> logits = torch.randn(2, 50, 7).softmax(dim=-1).log()
        >>> out = decoder(logits)
    """
    def __init__(
            self,
            labels: list,
            lm_path: str = None,
            alpha: int = 0,
            beta: int = 0,
            cutoff_top_n: int = 40,
            cutoff_prob: float = 1.0,
            beam_size: int = 3,
            num_processes: int = 4,
            blank_id: int = 0,
    ) -> None:
        super(BeamSearchCTC, self).__init__()
        try:
            from ctcdecode import CTCBeamDecoder  # TODO, need implement
        except:
            raise ImportError(CTCDECODE_IMPORT_ERROR)
        assert isinstance(labels, list), "labels must instance of list"
        self.decoder = CTCBeamDecoder(
            labels, lm_path, beta, cutoff_top_n, cutoff_prob, beam_size, num_processes, blank_id
        )

    def forward(self, logits: Tensor, sizes: Optional[Tensor] = None) -> Tensor:
        logits = logits.cpu()
        outputs, scores, offsets, seq_lens = self.decoder.decode(logits, sizes)
        return outputs
