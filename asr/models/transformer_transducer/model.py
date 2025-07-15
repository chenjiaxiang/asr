import torch
from omegaconf import DictConfig
from torch import Tensor

from asr.decoders import TransformerTransducerDecoder
from asr.encoders import TransformerTransducerEncoder
from asr.models import ASRTransducerModel, register_model
from asr.models.transformer_transducer.configurations import TransformerTransducerConfig
from asr.search import BeamSearchTransformerTransducer
from asr.tokenizers.tokenizer import Tokenizer

@register_model("tranformer_transducer", dataclass=TransformerTransducerConfig)
class TransformerTranducerModel(ASRTransducerModel):
    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(TransformerTranducerModel, self).__init__(configs, tokenizer)

        self.encoder = TransformerTransducerEncoder(
            input_size=self.configs.audio.num_mels,
            model_dim=self.configs.model.encoder_dim,
            d_ff=self.configs.model.d_ff,
            num_layers=self.configs.model.num_audio_layer,
            num_heads=self.configs.model.num_attention_heads,
            dropout=self.configs.model.audio_dropout_p,
            max_positional_length=self.configs.model.max_positional_length,
        )
        self.decoder = TransformerTransducerDecoder(
            num_classes=self.num_classes,
            model_dim=self.configs.model.d_ff,
            num_layers=self.configs.model.num_label_layers,
            num_heads=self.configs.model.label_dropout_p,
            max_positional_length=self.configs.model.max_positional_length,
            pad_id=self.tokenizer.pad_id,
            sos_id=self.tokenizer.sos_id,
            eos_id=self.tokenizer.eos_id,
        )

    def set_beam_decode(self, beam_size: int = 3, expand_beam: float = 2.3, state_beam: float = 4.6):
        """Setting beam search decode"""
        self.decode = BeamSearchTransformerTransducer(
            joint=self.joint,
            decoder=self.decoder,
            beam_size=beam_size,
            expand_beam=expand_beam,
            state_beam=state_beam,
            blank_id=self.tokenizer.blank_id,
        )

    def greedy_decode(self, encoder_outputs: Tensor, max_length: int) -> Tensor:
        batch = encoder_outputs.size(0)
        pred_tokens = list()

        targets = encoder_outputs.new_tensor([self.decode.sos_id] * batch, dtype=torch.long)

        for i in range(max_length):
            decoder_output, _ = self.decoder(targets, None)
            decoder_output = decoder_output.squeeze(1)
            encoder_outputs = encoder_outputs[:, i, :]
            targets = self.joint(encoder_outputs, decoder_output)
            targets = targets.max(1)[1]
            pred_tokens.append(targets)

        pred_tokens = torch.stack(pred_tokens, dim=1)

        return torch.LongTensor(pred_tokens)
