from omegaconf import DictConfig

from asr.decoders import LSTMAttentionDecoder
from asr.encoders import ConvolutionalLSTMEncoder, LSTMEncoder
from asr.models import ASREncoderDecoderModel, register_model
from asr.models.listen_attend_spell.configurations import (
    DeepCNNWiseJointCTCListenAttendSpellConfigs,
    JointCTCListenAttendSpellConfigs,
    ListenAttendSpellConfigs,
    ListenAttendSpellWithLocationAwareConfigs,
    ListenAttendSpellWithMultiHeadConfigs,
)
from asr.tokenizers.tokenizer import Tokenizer

@register_model("listen_atten_spell", dataclass=ListenAttendSpellConfigs)
class ListenAttendSpellModel(ASREncoderDecoderModel):
    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(ListenAttendSpellModel, self).__init__(configs, tokenizer)

        self.encoder = LSTMEncoder(
            input_dim=self.configs.audio.num_mels,
            num_layers=self.configs.model.num_encoder_layers,
            num_classes=self.num_classes,
            hidden_state_dim=self.configs.model.hidden_state_dim,
            dropout_p=self.configs.model.encoder_dropout_p,
            bidirectional=self.configs.encoder_bidirectional,
            rnn_type=self.configs.model.rnn_type,
            joint_ctc_attention=self.configs.model.joint_ctc_attention,
        )
        decoder_hidden_state_dim = (
            self.configs.model.hidden_state_dim << 1 if self.configs.model.encoder_bidirectional else self.configs.model.hidden_state_dim
        )
        self.decoder = LSTMAttentionDecoder(
            num_classes=self.num_classes,
            max_length=self.configs.model.max_length,
            hidden_state_dim=decoder_hidden_state_dim,
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


@register_model("listen_attend_spell_with_location_aware", dataclass=ListenAttendSpellWithLocationAwareConfigs)
class ListenAttendSpellWithLocationAwareModel(ASREncoderDecoderModel):
    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(ListenAttendSpellWithLocationAwareModel, self).__init__(configs, tokenizer)

        self.encoder = LSTMEncoder(
            input_dim=self.configs.audio.num_mels,
            num_layers=self.configs.model.num_encoder_layers,
            num_classes=self.num_classes,
            hidden_state_dim=self.configs.model.hidden_state_dim,
            dropout_p=self.configs.model.encoder_dropout_p,
            bidirectional=self.configs.encoder_bidirectional,
            rnn_type=self.configs.model.rnn_type,
            joint_ctc_attention=self.configs.model.joint_ctc_attention,
        )
        decoder_hidden_state_dim = (
            self.configs.model.hidden_state_dim << 1 if self.configs.model.encoder_bidirectional else self.configs.model.hidden_state_dim
        )
        self.decoder = LSTMAttentionDecoder(
            num_classes=self.num_classes,
            max_length=self.configs.model.max_length,
            hidden_state_dim=decoder_hidden_state_dim,
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

@register_model("listen_attend_spell_with_multi-head", dataclass=ListenAttendSpellWithMultiHeadConfigs)
class ListenAttendSpellWithMultiHeadModel(ASREncoderDecoderModel):
    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(ListenAttendSpellWithMultiHeadModel, self).__init__(configs, tokenizer)

        self.encoder = LSTMEncoder(
            input_dim=self.configs.audio.num_mels,
            num_layers=self.configs.model.num_encoder_layers,
            num_classes=self.num_classes,
            hidden_state_dim=self.configs.model.hidden_state_dim,
            dropout_p=self.configs.model.encoder_dropout_p,
            bidirectional=self.configs.encoder_bidirectional,
            rnn_type=self.configs.model.rnn_type,
            joint_ctc_attention=self.configs.model.joint_ctc_attention,
        )
        decoder_hidden_state_dim = (
            self.configs.model.hidden_state_dim << 1 if self.configs.model.encoder_bidirectional else self.configs.model.hidden_state_dim
        )
        self.decoder = LSTMAttentionDecoder(
            num_classes=self.num_classes,
            max_length=self.configs.model.max_length,
            hidden_state_dim=decoder_hidden_state_dim,
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

@register_model("joint_ctc_listen_attend_spell", dataclass=JointCTCListenAttendSpellConfigs)
class JointCTCListenAttendSpellModel(ASREncoderDecoderModel):
    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(JointCTCListenAttendSpellModel, self).__init__(configs, tokenizer)

        self.encoder = LSTMEncoder(
            input_dim=self.configs.audio.num_mels,
            num_layers=self.configs.model.num_encoder_layers,
            num_classes=self.num_classes,
            hidden_state_dim=self.configs.model.hidden_state_dim,
            dropout_p=self.configs.model.encoder_dropout_p,
            bidirectional=self.configs.encoder_bidirectional,
            rnn_type=self.configs.model.rnn_type,
            joint_ctc_attention=self.configs.model.joint_ctc_attention,
        )
        decoder_hidden_state_dim = (
            self.configs.model.hidden_state_dim << 1 if self.configs.model.encoder_bidirectional else self.configs.model.hidden_state_dim
        )
        self.decoder = LSTMAttentionDecoder(
            num_classes=self.num_classes,
            max_length=self.configs.model.max_length,
            hidden_state_dim=decoder_hidden_state_dim,
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

@register_model("deep_cnn_with_joint_ctc_listen_attend_spell", dataclass=DeepCNNWiseJointCTCListenAttendSpellConfigs)
class DeepCNNWithJointCTCListenAttendSpellModel(ASREncoderDecoderModel):
    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(DeepCNNWithJointCTCListenAttendSpellModel, self).__init__(configs, tokenizer)

        self.encoder = ConvolutionalLSTMEncoder(
            input_dim=self.configs.audio.num_mels,
            num_layers=self.configs.model.num_encoder_layers,
            num_classes=self.num_classes,
            hidden_state_dim=self.configs.model.hidden_state_dim,
            dropout_p=self.configs.model.encoder_dropout_p,
            bidirectional=self.configs.encoder_bidirectional,
            rnn_type=self.configs.model.rnn_type,
            joint_ctc_attention=self.configs.model.joint_ctc_attention,
        )
        decoder_hidden_state_dim = (
            self.configs.model.hidden_state_dim << 1 if self.configs.model.encoder_bidirectional else self.configs.model.hidden_state_dim
        )
        self.decoder = LSTMAttentionDecoder(
            num_classes=self.num_classes,
            max_length=self.configs.model.max_length,
            hidden_state_dim=decoder_hidden_state_dim,
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