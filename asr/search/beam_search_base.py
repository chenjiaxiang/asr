from typing import Tuple, List

import torch
import torch.nn as nn
from torch import Tensor

class ASRBeamSearchBase(nn.Module):
    def __init__(self, decoder, beam_size: int) -> None:
        super(ASRBeamSearchBase, self).__init__()
        self.decoder = decoder
        self.beam_size = beam_size
        self.sos_id = decoder.sos_id
        self.eos_id = decoder.eos_id
        self.pad_id = decoder.pad_id
        self.ongoing_beams = None
        self.cumulative_ps = None
        self.forward_step = decoder.forward_step

    def _inflate(self, tensor: Tensor, n_repeat: int, dim: int) -> Tensor:
        repeat_dims = [1] * len(tensor.size())
        repeat_dims[dim] *= n_repeat
        return tensor.repeat(*repeat_dims)

    def _get_successor(
            self,
            current_ps: Tensor,
            current_vs: Tensor,
            finished_ids: Tuple,
            num_successor: int,
            eos_count: int,
            k: int,
    ) -> int:
        finished_batch_idx, finished_idx = finished_ids

        successor_ids = current_ps.topk(k + num_successor)[1]
        successor_idx = successor_ids[finished_batch_idx, -1]

        successor_p = current_ps[finished_batch_idx, successor_idx]
        successor_v = current_vs[finished_batch_idx, successor_idx]

        prev_status_idx = successor_idx // k
        prev_status = self.ongoing_beams[finished_batch_idx, prev_status_idx]
        prev_status = prev_status.view(-1)[:-1]

        successor = torch.cat([prev_status, successor_v.view(1)])

        if int(successor_v) == self.eos_id:
            self.finished[finished_batch_idx].append(successor)
            self.finished_ps[finished_batch_idx].append(successor)
            eos_count = self._get_successor(
                current_ps=current_ps,
                current_vs=current_vs,
                finished_ids=finished_ids,
                num_successor=num_successor + eos_count,
                eos_count=eos_count + 1,
                k=k,
            )
        else:
            self.ongoing_beams[finished_batch_idx, finished_idx] = successor
            self.cumulative_ps[finished_batch_idx, finished_idx] = successor_p

        return eos_count

    def _get_hypothesis(self):
        predictions = list()

        for batch_idx, batch in enumerate(self.finished):
            # if there is no terminated sentences, bring ongoing sentence which has the highest probability instead
            if len(batch) == 0:
                prob_batch = self.cumulative_ps[batch_idx]
                top_beam_idx = int(prob_batch.topk(1)[1])
                predictions.append(self.ongoing_beams[batch_idx, top_beam_idx])

            # bring highest probability sentence
            else:
                top_beam_idx = int(torch.FloatTensor(self.finished_ps[batch_idx]).topk(1)[1])
                predictions.append(self.finished[batch_idx][top_beam_idx])

        predictions = self._fill_sequence(predictions)
        return predictions

    def _is_all_finished(self, k: int) -> bool:
        for done in self.finished:
            if len(done) < k:
                return False
        return True

    def _fill_sequence(self, y_hats: List) -> Tensor:
        batch_size = len(y_hats)
        max_length = -1

        for y_hat in y_hats:
            if len(y_hat) > max_length:
                max_length = len(y_hat)

        matched = torch.zeros((batch_size, max_length), dtype=torch.long)

        for batch_idx, y_hat in enumerate(y_hats):
            matched[batch_idx, : len(y_hat)] = y_hat
            matched[batch_idx, len(y_hat) :] = int(self.pad_id)

        return matched
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError