from dataclasses import dataclass, field
from asr.dataclass.configurations import ASRDataclass


@dataclass
class TransformerTransducerConfig(ASRDataclass):
    model_name: str = field(default="transfomer_transducer", metadata={"help": "Model Name."})
    encoder_dim: int = field(default=512, metadata={"help": "Dimension of encoder name."})
    d_ff: int = field(default=2048, metadata={"help": "Dimension of feed forward network."})
    num_audio_layers: int = field(default=18, metadata={"help": "Number of audio layers."})
    num_label_layers: int = field(default=2, metadata={"help": "Number of label layers."})
    num_attention_heads: int = field(default=8, metadata={"help": "The number of attention heads."})
    audio_dropout_p: float = field(default=0.1, metadata={"help": "Dropout probability of audio layer."})
    label_dropout_p: float = field(default=0.1, metadata={"help": "Dropout probability of label layer."})
    decoder_hidden_state_dim: int = field(default=512, metadata={"help": "Hidden state dimension of decoder."})
    decoder_output_dim: int = field(default=512, metadata={"help": "Dimension of model output."})
    conv_kernel_size: int = field(default=31, metadata={"help": "Kernel size of convolution layer."})
    max_positional_lengths: int = field(default=5000, metadata={"help": "Max length of positional encoding."})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})