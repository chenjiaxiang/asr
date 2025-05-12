from collections import OrderedDict
from typing import Dict

from omegaconf import DictConfig
from torch import Tensor

from asr.models import ASRModel
from asr.tokenizers.tokenizer import Tokenizer
from asr.utils import get_class_name


class ASREncoderDecoderModel(ASRModel):
    def __init__(
            self,
            configs: DictConfig,
            tokenizer: Tokenizer,
    ) -> None:
        super(ASREncoderDecoderModel, self).__init__(configs, tokenizer)
        self.teacher_forcing_ratio = configs.model.teacher_forcing_ratio
        self.encoder = None
        self.decoder = None

    def set_beam_decoder(self, beam_size = 3):
        raise NotImplementedError

    def collect_outputs(
            self,
            stage: str,
            logits: Tensor,
            encoder_logits: Tensor,
            encoder_output_lengths: Tensor,
            targets: Tensor,
            target_lengths: Tensor
    ) -> OrderedDict:
        cross_entropy_loss, ctc_loss = None, None

        if get_class_name(self.criterion) == "JointCTCCrossEntropyLoss":
            loss, ctc_loss, cross_entropy_loss = self.criterion(
                encoder_logits=encoder_logits.transpose(0, 1),
                logits=logits,
                output_lengths=encoder_output_lengths,
                targets=targets[:, 1:],
                target_lengths=target_lengths,
            )
            self.info(
                {
                    f"{stage}_loss": loss,
                    f"{stage}_cross_entropy_loss": cross_entropy_loss,
                    f"{stage}_ctc_loss": ctc_loss,
                }
            )
        elif (
            get_class_name(self.criterion) == "LabelSmoothedCrossEntropyLoss"
            or get_class_name(self.criterion) == "CrossEntropyLoss"
        ):
            loss = self.criterion(logits, targets[:, 1:])
            self.info({f"{stage}_loss": loss})
        else:
            raise ValueError(f"Unsupported criterion: {self.criterion}")

        predictions = logits.max(-1)[1]

        wer = self.wer_metric(targets[:, 1:], predictions)
        cer = self.cer_metric(targets[:, 1:], predictions)

        self.info(
            {
                f"{stage}_wer": wer,
                f"{stage}_cer": cer,
            }
        )

        return OrderedDict(
            {
                "loss": loss,
                "cross_entropy_loss": cross_entropy_loss,
                "ctc_loss": ctc_loss,
                "predictions": predictions,
                "targets": targets,
                "logits": logits,
                "learning_rata": self.get_lr(),
            }
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Dict[str, Tensor]:
        logits = None
        encoder_outputs, encoder_logits, encoder_output_lengths = self.encoder(inputs, input_lengths)

        if get_class_name(self.decoder) in ("BeamSearchLSTM", "BeamSearchTransformer"):
            predictions = self.decoder(encoder_outputs, encoder_output_lengths)
        else:
            logits = self.decoder(
                encoder_outputs=encoder_outputs,
                encoder_output_lengths=encoder_output_lengths,
                teacher_forcing_ratio=0.0,
            )
            predictions = logits.max(-1)[1]
        return {
            "predictions": predictions,
            "logits": logits,
            "encoder_outputs": encoder_outputs,
            "encoder_logits": encoder_logits,
            "encoder_output_lengths": encoder_output_lengths,
        }