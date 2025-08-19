from collections import OrderedDict
from typing import Dict

import torch
from omegaconf import DictConfig

from asr.models import ASRModel
from asr.tokenizers.tokenizer import Tokenizer

class ASRCTCModel(ASRModel):
    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None: 
        super(ASRCTCModel, self).__init__(configs, tokenizer)
        self.encoder = None
        self.decoder = None

    def set_beam_decoder(self, beam_size: int = 3):
        """Setting beam search decoder"""
        from asr.search import BeamSearchCTC
        
        self.decoder = BeamSearchCTC(
            labels=self.tokenizer.labels,
            blank_id=self.tokenizer.blank_id,
            beam_size=beam_size,
        )

    def collect_outputs(
            self,
            stage: str,
            logits: torch.FloatTensor,
            output_lengths: torch.IntTensor,
            targets: torch.IntTensor,
            target_lengths: torch.IntTensor,
    ) -> OrderedDict:
        loss = self.criterion(
            log_probs=logits.transpose(0, 1),
            targets=targets[:, 1:],
            input_lengths=output_lengths,
            target_lengths=target_lengths,
        )
        predictions = logits.max(-1)[1]

        wer = self.wer_metric(targets[:, 1:], predictions)
        cer = self.cer_metric(targets[:, 1:], predictions)

        self.info(
            {
                f"{stage}_wer": wer,
                f"{stage}_cer": cer,
                f"{stage}_loss": loss,
                "learning_rate": self.get_lr(),
            }
        )

        return OrderedDict(
            {
                "loss": loss,
                "wer": wer,
                "cer": cer,
            }
        )

    def forward(self, inputs: torch.FloatTensor, input_lengths: torch.IntTensor) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(inputs, input_lengths)

        if len(outputs) == 2:
            logits, output_lengths = outputs
        else:
            logits, _, output_lengths = outputs

        logits = self.fc(logits).log_softmax(dim=-1)

        if self.decoder is not None:
            y_hats = self.decoder(logits)
        else:
            y_hats = logits.max(-1)[1]
        
        return {
            "predictions": y_hats,
            "logits": logits,
            "output_lengths": output_lengths,
        }

    def training_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        inputs, targets, input_lengths, target_lengths = batch
        logits, output_lengths = self.encoder(inputs, input_lengths)
        return self.collect_outputs(
            stage="train",
            logits=logits,
            output_lengths=output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

    def validation_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        inputs, targets, input_lengths, target_lengths = batch
        logits, output_lengths = self.encoder(inputs, input_lengths)
        return self.collect_outputs(
            stage="val",
            logits=logits,
            output_lengths=output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

    def validation_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        inputs, targets, input_lengths, target_lengths = batch
        logits, output_lengths = self.encoder(inputs, input_lengths)
        return self.collect_outputs(
            stage="test",
            logits=logits,
            output_lengths=output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )