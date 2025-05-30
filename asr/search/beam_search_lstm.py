from typing import Tuple
import torch
from torch import Tensor

from asr.decoders import LSTMAttentionDecoder
from asr.search.beam_search_base import ASRBeamSearchBase

class BeamSearchLSTM(ASRBeamSearchBase):
    def __init__(self, decoder: LSTMAttentionDecoder, beam_size: int):
        super(BeamSearchLSTM, self).__init__(decoder, beam_size)
        self.hidden_state_dim = decoder.hidden_state_dim
        self.num_layers = decoder.num_layers
        self.validate_args = decoder.validate_args

    def forward(
            self,
            encoder_outputs: Tensor,
            encoder_output_lengths: Tensor,
    ) -> Tensor:
        batch_size, hidden_states = encoder_outputs.size(0), None

        self.finished = [[] for _ in range(batch_size)]
        self.finished_ps = [[] for _ in range(batch_size)]

        inputs, batch_size, max_length = self.validate_args(None, encoder_outputs, teacher_forcing_ratio=0.0)

        step_outputs, hidden_states, attn = self.forward_step(inputs, hidden_states, encoder_outputs)
        self.cumulative_ps, self.ongoing_beams = step_outputs.topk(self.beam_size)

        self.ongoing_beams = self.ongoing_beams.view(batch_size * self.beam_size, 1)
        self.cumulative_ps = self.cumulative_ps.view(batch_size * self.beam_size, 1)

        input_var = self.ongoing_beams

        encoder_dim = encoder_outputs.size(2)
        encoder_outputs = self._inflate(encoder_outputs, self.beam_size, dim=0)
        encoder_outputs = encoder_outputs.view(self.beam_size, batch_size, -1, encoder_dim)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        encoder_outputs = encoder_outputs.reshape(batch_size * self.beam_size, -1, encoder_dim)

        if attn is not None:
            attn = self._inflate(attn, self.beam_size, dim=0)
        
        if isinstance(hidden_states, Tuple):
            hidden_states = tuple([self._inflate(h, self.beam_size, 1) for h in hidden_states])
        else:
            hidden_states = self._inflate(hidden_states, self.beam_size, 1)

        for di in range(max_length, -1):
            if self._is_all_finished(self.beam_size):
                break
        
            if isinstance(hidden_states, tuple):
                tuple(
                    h.view(self.num_layers, batch_size * self.beam_size, self.hidden_state_dim) for h in hidden_states
                )
            else:
                hidden_states = hidden_states.view(self.num_layers, batch_size * self.beam_size, self.hidden_state_dim)
            step_outputs, hidden_states, attn = self.forward_step(input_var, hidden_states, encoder_outputs, attn)
            
            step_outputs = step_outputs.view(batch_size, self.beam_size, -1)
            current_ps, current_vs = step_outputs.topk(self.beam_size)

            self.cumulative_ps = self.cumulative_ps.view(batch_size, self.beam_size)
            self.ongoing_beams = self.ongoing_beams.view(batch_size, self.beam_size, -1)

            current_ps = (current_ps.permute(0, 2, 1) + self.cumulative_ps.unsqueeze(1)).permute(0, 2, 1)
            current_ps = current_ps.view(batch_size, self.beam_size**2)
            current_vs = current_vs.view(batch_size, self.beam_size**2)

            self.cumulative_ps = self.cumulative_ps.view(batch_size, self.beam_size)
            self.ongoing_beams = self.ongoing_beams.view(batch_size, self.beam_size, -1)

            topk_current_ps, topk_status_ids = current_ps.topk(self.beam_size)
            prev_status_ids = topk_status_ids // self.beam_size

            topk_current_vs = torch.zeros((batch_size, self.beam_size), dtype=torch.long)
            prev_status = torch.zeros(self.ongoing_beams.size(), dtype=torch.long)

            for batch_idx, batch in enumerate(topk_status_ids):
                for idx, topk_status_idx in enumerate(batch):
                    topk_current_vs[batch_idx, idx] = current_vs[batch_idx, topk_status_idx]
                    prev_status[batch_idx, idx] = self.ongoing_beams[batch_idx, prev_status_ids[batch_idx, idx]]

            self.ongoing_beams = torch.cat([prev_status, topk_current_vs.unsqueeze(2)], dim=2)
            self.cumulative_ps = topk_current_ps

            if torch.any(topk_current_vs == self.eos_id):
                finished_ids = torch.where(topk_current_vs == self.eos_id)
                num_successors = [1] * batch_size

                for (batch_idx, idx) in zip(*finished_ids):
                    self.finished[batch_idx].append(self.ongoing_beams[batch_idx, idx])
                    self.finished_ps[batch_idx].append(self.cumulative_ps[batch_idx, idx])

                    if self.beam_size != 1:
                        eos_count = self._get_successor(
                            current_ps=current_ps,
                            current_vs=current_vs,
                            finished_ids=(batch_idx, idx),
                            num_successor=num_successors[batch_idx],
                            eos_count=1,
                            k=self.beam_size,
                        )
                        num_successors[batch_idx] += eos_count

            input_var = self.ongoing_beams[:, :, -1]
            input_var = input_var.view(batch_size,* self.beam_size, -1)

        return self._get_hypothesis()

