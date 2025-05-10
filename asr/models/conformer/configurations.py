from dataclasses import dataclass, field
from asr.dataclass.configurations import ASRDataclass


@dataclass
class ConformerConfigs(ASRDataclass):
    model_name: str = field(default="conformer", metadata={"help": "Model name"})
    encoder_dim: int = field(default=512, metadata={"help": "Dimension of encoder"})
    num_encoder_layers: int = field(default=17, metadata={"help": "The number of encoder layers."})
    num_attention_heads: int = field(default=8, metadata={"help": "The number of attention heads."})
    feed_forward_expansion_factor: int = field(
        default=4, metadata={"help": "The expansion factor of feed forward module."}
    )
    conv_expansion_factor: int = field(default=2, metadata={"help": "The expansion factor of convolution module."})
    input_dropout_p: float = field(default=0.1, metadata={"help": "The dropout probability of inputs."})
    feed_forward_dropout_p: float = field(
        default=0.1, metadata={"help": "The dropout probability of feed forward module."}
    )
    attention_dropout_p: float = field(default=0.1, metadata={"help": "The dropout probability of attention module."})
    conv_dropout_p: float = field(default=0.1, metadata={"help": "The dropout probability of convolution module."})
    conv_kernel_size: int = field(default=31, metadata={"help": "The kernel size of convolution."})
    half_step_residual: bool = field(
        default=True, metadata={"help": "Flag indication whether to use half step residual or not."}
    )
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})

@dataclass
class ConformerLSTMConfigs(ASRDataclass):
    model_name: str = field(default="conformer_lstm", metadata={"help": "Model name"})
    encoder_dim: int = field(default=512, metadata={"help": "Dimension of encoder"})
    num_encoder_layers: int = field(default=17, metadata={"help": "The number of encoder layers."})
    num_attention_heads: int = field(default=8, metadata={"help": "The number of attention heads."})
    feed_forward_expansion_factor: int = field(
        default=4, metadata={"help": "The expansion factor of feed forward module."}
    )
    conv_expansion_factor: int = field(default=2, metadata={"help": "The expansion factor of convolution module."})
    input_dropout_p: float = field(default=0.1, metadata={"help": "The dropout probability of inputs."})
    feed_forward_dropout_p: float = field(
        default=0.1, metadata={"help": "The dropout probability of feed forward module."}
    )
    attention_dropout_p: float = field(default=0.1, metadata={"help": "The dropout probability of attention module."})
    conv_dropout_p: float = field(default=0.1, metadata={"help": "The dropout probability of convolution module."})
    conv_kernel_size: int = field(default=31, metadata={"help": "The kernel size of convolution."})
    half_step_residual: bool = field(
        default=True, metadata={"help": "Flag indication whether to use half step residual or not."}
    )
    num_decoder_layers: int = field(default=2, metadata={"help": "The number of decoder layers."})
    decoder_dropout_p: float = field(default=0.1, metadata={"help": "The dropout probability of decoder."})
    max_length: int = field(default=128, metadata={"help": "Max decoding length."})
    teacher_forcing_ratio: float = field(default=1.0, metadata={"help": "The ratio of teacher forcing."})
    rnn_type: str = field(default="lstm", metadata={"help": "Type of rnn cell (rnn, lstm, gru)"})  # TODO fix
    decoder_attn_mechanism: str = field(default="loc", metadata={"help": "The attention mechanism for decoder."})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})

