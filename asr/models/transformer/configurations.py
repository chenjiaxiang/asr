from dataclasses import dataclass, field
from asr.dataclass.configurations import ASRDataclass

@dataclass
class TransformerConfigs(ASRDataclass):
    model_name: str = field(default="transfomer", metadata={"help": "Model Name."})
    d_model: int = field(default=512, metadata={"help": "Dimension of model."})
    d_ff: int = field(default=2048, metadata={"help": "Dimension of feed forward network."})
    num_attention_heads: int = field(default=8, metadata={"help": "The number of attention heads."})
    num_encoder_layers: int = field(default=12, metadata={"help": "The number of encoder layers."})
    num_decoder_layers: int = field(default=6, metadata={"help": "The number of decoder layers."})
    encoder_drouput_p: float = field(default=0.3, metadata={"help": "The dropout probability of encoder."})
    decoder_dropout_p: float = field(default=0.3, metadata={"help": "The dropout probability of decoder."})
    ffnet_style: str = field(default="ff", metadata={"help": "Style of feed forward network. (ff, conv)"})
    max_length: int = field(default=128, metadata={"help": "Max decoding length."})
    teacher_forcing_ratio: float = field(default=1.0, metadata={"help": "The ratio of teacher forcing."})
    joint_ctc_attention: bool = field(default=False, metadata={"help": "Flag indication joint ctc attention or not."})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})


@dataclass
class JointCTCTransformerConfigs(ASRDataclass):
    model_name: str = field(default="joint_ctc_transfomer", metadata={"help": "Model Name."})
    extractot: str = field(default="conv2d_subsample", metadata={"help": "The CNN feature extractor."})
    d_model: int = field(default=512, metadata={"help": "Dimension of model."})
    d_ff: int = field(default=2048, metadata={"help": "Dimension of feed forward network."})
    num_attention_heads: int = field(default=8, metadata={"help": "The number of attention heads."})
    num_encoder_layers: int = field(default=12, metadata={"help": "The number of encoder layers."})
    num_decoder_layers: int = field(default=6, metadata={"help": "The number of decoder layers."})
    encoder_drouput_p: float = field(default=0.3, metadata={"help": "The dropout probability of encoder."})
    decoder_dropout_p: float = field(default=0.3, metadata={"help": "The dropout probability of decoder."})
    ffnet_style: str = field(default="ff", metadata={"help": "Style of feed forward network. (ff, conv)"})
    max_length: int = field(default=128, metadata={"help": "Max decoding length."})
    teacher_forcing_ratio: float = field(default=1.0, metadata={"help": "The ratio of teacher forcing."})
    joint_ctc_attention: bool = field(default=False, metadata={"help": "Flag indication joint ctc attention or not."})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})


@dataclass
class TransformerWithCTCConfigs(ASRDataclass):
    model_name: str = field(default="transfomer_with_ctc", metadata={"help": "Model Name."})
    d_model: int = field(default=512, metadata={"help": "Dimension of model."})
    d_ff: int = field(default=2048, metadata={"help": "Dimension of feed forward network."})
    num_attention_heads: int = field(default=8, metadata={"help": "The number of attention heads."})
    num_encoder_layers: int = field(default=12, metadata={"help": "The number of encoder layers."})
    encoder_drouput_p: float = field(default=0.3, metadata={"help": "The dropout probability of encoder."})
    ffnet_style: str = field(default="ff", metadata={"help": "Style of feed forward network. (ff, conv)"})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})


@dataclass
class VGGTransformerConfigs(ASRDataclass):
    model_name: str = field(default="vgg_transfomer", metadata={"help": "Model Name."})
    extractot: str = field(default="vgg", metadata={"help": "The CNN feature extractor."})
    d_model: int = field(default=512, metadata={"help": "Dimension of model."})
    d_ff: int = field(default=2048, metadata={"help": "Dimension of feed forward network."})
    num_attention_heads: int = field(default=8, metadata={"help": "The number of attention heads."})
    num_encoder_layers: int = field(default=12, metadata={"help": "The number of encoder layers."})
    num_decoder_layers: int = field(default=6, metadata={"help": "The number of decoder layers."})
    encoder_drouput_p: float = field(default=0.3, metadata={"help": "The dropout probability of encoder."})
    decoder_dropout_p: float = field(default=0.3, metadata={"help": "The dropout probability of decoder."})
    ffnet_style: str = field(default="ff", metadata={"help": "Style of feed forward network. (ff, conv)"})
    max_length: int = field(default=128, metadata={"help": "Max decoding length."})
    teacher_forcing_ratio: float = field(default=1.0, metadata={"help": "The ratio of teacher forcing."})
    joint_ctc_attention: bool = field(default=False, metadata={"help": "Flag indication joint ctc attention or not."})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})