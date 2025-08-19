from dataclasses import dataclass, field

from asr.dataclass.configurations import ASRDataclass


@dataclass
class ListenAttendSpellConfigs(ASRDataclass):
    model_name: str = field(default="listen_attend_spell", metadata={"help": "Model name"})
    num_encoder_layers: int = field(default=3, metadata={"help": "The number of encoder layers."})
    num_decoder_layers: int = field(default=2, metadata={"help": "The number of decoder layers."})
    hidden_state_dim: int = field(default=512, metadata={"help": "The hidden state dimension of encoder."})
    encoder_dropout_p: float = field(default=0.3, metadata={"help": "The dropout probability of encoder."})
    encoder_bidirectional: bool = field(default=True, metadata={"help": "If True, becomes a bidirectional encoders."})
    rnn_type: str = field(default="lstm", metadata={"help": "Type of rnn cell (rnn, lstm, gru)"})
    joint_ctc_attention: bool = field(default=False, metadata={"help": "Flag indication joint ctc attention or not."})
    max_length: int = field(default=128, metadata={"help": "Max decoding length."})
    num_attention_heads: int = field(default=1, metadata={"help": "The number of attention heads.`"})
    decoder_dropout_p: float = field(default=0.2, metadata={"help": "The dropout probability of decoder."})
    decoder_attn_mechanism: str = field(default="dot", metadata={"help": "The attention mechanism for decoder."})
    teacher_forcing_ratio: float = field(default=1.0, metadata={"help": "The ratio of teacher forcing."})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})

@dataclass
class ListenAttendSpellWithLocationAwareConfigs(ASRDataclass):
    model_name: str = field(default="listen_attend_spell_with_location_aware", metadata={"help": "Model name"})
    num_encoder_layers: int = field(default=3, metadata={"help": "The number of encoder layers."})
    num_decoder_layers: int = field(default=2, metadata={"help": "The number of decoder layers."})
    hidden_state_dim: int = field(default=512, metadata={"help": "The hidden state dimension of encoder."})
    encoder_dropout_p: float = field(default=0.3, metadata={"help": "The dropout probability of encoder."})
    encoder_bidirectional: bool = field(default=True, metadata={"help": "If True, becomes a bidirectional encoders."})
    rnn_type: str = field(default="lstm", metadata={"help": "Type of rnn cell (rnn, lstm, gru)"})
    joint_ctc_attention: bool = field(default=False, metadata={"help": "Flag indication joint ctc attention or not."})
    max_length: int = field(default=128, metadata={"help": "Max decoding length."})
    num_attention_heads: int = field(default=1, metadata={"help": "The number of attention heads.`"})
    decoder_dropout_p: float = field(default=0.2, metadata={"help": "The dropout probability of decoder."})
    decoder_attn_mechanism: str = field(default="loc", metadata={"help": "The attention mechanism for decoder."})
    teacher_forcing_ratio: float = field(default=1.0, metadata={"help": "The ratio of teacher forcing."})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})

@dataclass
class ListenAttendSpellWithMultiHeadConfigs(ASRDataclass):
    model_name: str = field(default="listen_attend_spell_with_multi_head", metadata={"help": "Model name"})
    num_encoder_layers: int = field(default=3, metadata={"help": "The number of encoder layers."})
    num_decoder_layers: int = field(default=2, metadata={"help": "The number of decoder layers."})
    hidden_state_dim: int = field(default=512, metadata={"help": "The hidden state dimension of encoder."})
    encoder_dropout_p: float = field(default=0.3, metadata={"help": "The dropout probability of encoder."})
    encoder_bidirectional: bool = field(default=True, metadata={"help": "If True, becomes a bidirectional encoders."})
    rnn_type: str = field(default="lstm", metadata={"help": "Type of rnn cell (rnn, lstm, gru)"})
    joint_ctc_attention: bool = field(default=False, metadata={"help": "Flag indication joint ctc attention or not."})
    max_length: int = field(default=128, metadata={"help": "Max decoding length."})
    num_attention_heads: int = field(default=1, metadata={"help": "The number of attention heads.`"})
    decoder_dropout_p: float = field(default=0.2, metadata={"help": "The dropout probability of decoder."})
    decoder_attn_mechanism: str = field(default="multi-head", metadata={"help": "The attention mechanism for decoder."})
    teacher_forcing_ratio: float = field(default=1.0, metadata={"help": "The ratio of teacher forcing."})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})

@dataclass
class JointCTCListenAttendSpellConfigs(ASRDataclass):
    model_name: str = field(default="joint_ctc_listen_attend_spell", metadata={"help": "Model name"})
    num_encoder_layers: int = field(default=3, metadata={"help": "The number of encoder layers."})
    num_decoder_layers: int = field(default=2, metadata={"help": "The number of decoder layers."})
    hidden_state_dim: int = field(default=768, metadata={"help": "The hidden state dimension of encoder."})
    encoder_dropout_p: float = field(default=0.3, metadata={"help": "The dropout probability of encoder."})
    encoder_bidirectional: bool = field(default=True, metadata={"help": "If True, becomes a bidirectional encoders."})
    rnn_type: str = field(default="lstm", metadata={"help": "Type of rnn cell (rnn, lstm, gru)"})
    joint_ctc_attention: bool = field(default=True, metadata={"help": "Flag indication joint ctc attention or not."})
    max_length: int = field(default=128, metadata={"help": "Max decoding length."})
    num_attention_heads: int = field(default=1, metadata={"help": "The number of attention heads.`"})
    decoder_dropout_p: float = field(default=0.2, metadata={"help": "The dropout probability of decoder."})
    decoder_attn_mechanism: str = field(default="loc", metadata={"help": "The attention mechanism for decoder."})
    teacher_forcing_ratio: float = field(default=1.0, metadata={"help": "The ratio of teacher forcing."})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})

@dataclass
class DeepCNNWiseJointCTCListenAttendSpellConfigs(ASRDataclass):
    model_name: str = field(default="deep_cnn_with_joint_ctc_listen_attend_spell", metadata={"help": "Model name"})
    num_encoder_layers: int = field(default=3, metadata={"help": "The number of encoder layers."})
    num_decoder_layers: int = field(default=2, metadata={"help": "The number of decoder layers."})
    hidden_state_dim: int = field(default=768, metadata={"help": "The hidden state dimension of encoder."})
    encoder_dropout_p: float = field(default=0.3, metadata={"help": "The dropout probability of encoder."})
    encoder_bidirectional: bool = field(default=True, metadata={"help": "If True, becomes a bidirectional encoders."})
    rnn_type: str = field(default="lstm", metadata={"help": "Type of rnn cell (rnn, lstm, gru)"})
    extractor: str = field(default="vgg", metadata={"help": "The CNN feature extractor."})
    activation: str = field(default="headtanh", metadata={"help": "Type if activation function."})
    joint_ctc_attention: bool = field(default=True, metadata={"help": "Flag indication joint ctc attention or not."})
    max_length: int = field(default=128, metadata={"help": "Max decoding length."})
    num_attention_heads: int = field(default=1, metadata={"help": "The number of attention heads.`"})
    decoder_dropout_p: float = field(default=0.2, metadata={"help": "The dropout probability of decoder."})
    decoder_attn_mechanism: str = field(default="loc", metadata={"help": "The attention mechanism for decoder."})
    teacher_forcing_ratio: float = field(default=1.0, metadata={"help": "The ratio of teacher forcing."})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})

