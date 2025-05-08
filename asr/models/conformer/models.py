from omegaconf import DictConfig
from asr.encoders import ConformerEncoder
from asr.decoders import LSTMAttentionDecoder
from asr.models import (
    register_model,
)
from asr.models import (
    ASREncoderDecoderModel,
)
from asr.models.conformer.configurations import (
    ConformerLSTMConfigs,
)
from asr.tokenizers.tokenizer import Tokenizer

@register_model("conformer_lstm", datacalss=ConformerLSTMConfigs)
class ConformerLSTMModel(ASREncoderDecoderModel):
    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(ConformerLSTMModel, self).__init__(configs, tokenizer)

        self.encoder = ConformerEncoder(
            num_classes=self.num_classes,
            input_dim=self.configs.audio.num_mels,
            encoder_dim=self.configs.model.encoder_dim,
            num_layers=self.configs.model.num_encoder_layers,
            num_attention_heads=self.configs.model.num_attention_heads,
            feed_forward_expansion_factor=self.configs.model.feed_forward_expansion_factor,
            conv_expansion_factor=self.configs.model.conv_expansion_factor,
            input_dropout_p=self.configs.model.input_dropout_p,
            feed_forward_dropout_p=self.configs.model.feed_forward_dropout_p,
            attention_dropout_p=self.configs.model.attention_dropout_p,
            conv_dropout_p=self.configs.model.conv_dropout_p,
            conv_kernel_size=self.configs.model.conv_kernel_size,
            half_step_residual=self.configs.model.half_step_residual,
            joint_ctc_attention=False,
        )
        self.decoder = LSTMAttentionDecoder(
            num_classes=self.num_classes,
            max_lengths=self.configs.model.max_length,
            hidden_state_dim=self.configs.model.encoder_dim,
            pad_id=self.tokenizer.pad_id,
            sos_id=self.tokenizer.sos_id,
            eos_id=self.tokenizer.eos_id,
            num_heads=self.configs.model.num_attention_heads,
            dropout_p=self.configs.model.decoder_dropout_p,
            num_layers=self.configs.model.num_decoder_layers,
            attn_mechanism=self.configs.model.decoder_attn_mechanism,
            rnn_type=self.configs.model.rnn_type,
        )

    def set_beam_decoder(self, beam_size: int = 3):
        from asr.search import BeamSearchLSTM

        self.decoder = BeamSearchLSTM(
            decoder=self.decoder,
            beam_size=beam_size,
        )