import torch.nn as nn

from asr.utils import CTCDECODE_IMPORT_ERROR


class BeamSearchCTC(nn.Module):
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
            from ctcdecode import CTCBeamDecoder # TODO, need implement
        except:
            raise ImportError(CTCDECODE_IMPORT_ERROR)
        assert isinstance(labels, list), "labels must instance of list"
        self.decoder = CTCBeamDecoder(
            labels, lm_path, beta, cutoff_top_n, cutoff_prob, beam_size, num_processes, blank_id
        )
    
    def forward(self, logits, sizes=None):
        logits = logits.cpu()
        outputs, scores, offsets, seq_lens = self.decoder.decode(logits, sizes)
        return outputs