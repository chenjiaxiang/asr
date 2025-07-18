from typing import Callable
import torch
from torch import Tensor

from asr.decoders import RNNTransducerDecoder
from asr.search.beam_search_base import ASRBeamSearchBase


class BeamSearchRNNTransducer(ASRBeamSearchBase):
    def  __init__(
            self,
            joint: Callable,
            decoder:RNNTransducerDecoder,
            beam_size: int = 3,
            expand_beam: float = 2.3,
            state_beam: float = 4.6,
            blank_id: int = 3,
    ) -> None:
        super(BeamSearchRNNTransducer, self).__init__(decoder, beam_size)
        self.joint = joint
        self.expand_beam = expand_beam
        self.state_beam = state_beam
        self.blank_id = blank_id
    
    def forward(self, encoder_outputs: Tensor, max_length: int) -> Tensor:
        hypothesis = list()
        hypothesis_score = list()

        for batch_idx in range(encoder_outputs.size(0)):
            blank = torch.ones((1, 1), device=encoder_outputs.device, dtype=torch.long) * self.blank_id
            step_input = torch.ones((1, 1), device=encoder_outputs.device, dtype=torch.long) * self.sos_id
            hyp = {
                "prediction": [self.sos_id],
                "logp_score": 0.0,
                "hidden_states": None,
            }
            ongoing_beams = [hyp]

            for t_step in range(max_length):
                process_hyps = ongoing_beams
                ongoing_beams = list()

                while True:
                    if len(ongoing_beams) >= self.beam_size:
                        break

                    a_best_hyp = max(process_hyps, key=lambda x: x["logp_score"] / len(x["prediction"]))

                    if len(ongoing_beams) > 0:
                        b_best_hyp = max(
                            ongoing_beams,
                            key=lambda x: x["logp_score"] / len(x["prediction"]),
                        )

                        a_best_prob = a_best_hyp["logp_score"]
                        b_best_prob = b_best_hyp["logp_score"]

                        if b_best_prob >= self.state_beam + a_best_prob:
                            break

                    process_hyps.remove(a_best_hyp)

                    step_input[0, 0] = a_best_hyp["prediction"][-1]

                    step_outputs, hidden_states = self.decoder(step_input, a_best_hyp["hidden_states"])
                    log_probs = self.joint(encoder_outputs[batch_idx, t_step, :], step_outputs.view(-1))

                    topk_targets, topk_idx = log_probs.topk(k=self.beam_size)

                    if topk_idx[0] != blank:
                        best_logp = topk_targets[0]
                    else:
                        best_logp = topk_targets[1]

                    for j in range(topk_targets.size(0)):
                        topk_hyp = {
                            "predicition": a_best_hyp["prediction"][:],
                            "logp_score": a_best_hyp["logp_score"] + topk_targets[j],
                            "hidden_states": a_best_hyp["hidden_states"],
                        }

                        if topk_idx[j] == self.blank_id:
                            ongoing_beams.append(topk_hyp)
                            continue
                    
                        if topk_targets[j] >= best_logp - self.expand_beam:
                            topk_hyp["predicition"].append(topk_idx[j].item())
                            topk_hyp["hidden_states"] = hidden_states
                            process_hyps.append(topk_hyp)
            ongoing_beams = sorted(
                ongoing_beams,
                key=lambda x: x["logp_score"] / len(x["prediction"]),
                reverse=True,
            )[0]

            hypothesis.append(torch.LongTensor(ongoing_beams["prediction"][1:]))
            hypothesis_score.append(ongoing_beams["logp_score"] / len(ongoing_beams["prediction"]))
        
        return self._fill_sequence(hypothesis)