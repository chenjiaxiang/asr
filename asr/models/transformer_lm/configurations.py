from dataclasses import dataclass, field

from asr.dataclass.configurations import ASRDataclass

@dataclass
class TransformerLanguageModelConfigs(ASRDataclass):
    model_name: str = field(default="transformer_lm", metadata={"help": "Model name"})
    num_layers: int = field(default=6, metadata={"help": "The number of encoder layers."})
    d_model: int = field(default=768, metadata={"help": "The dimension of model."})
    d_ff: int = field(default=1536, metadata={"help": "The dimension of feed forward network."})
    num_attention_heads: int = field(default=8, metadata={"help": "The number of attention heads."})
    dropout_p: float = field(default=0.3, metadata={"help": "The dropout probability of encoder."})
    max_length: int = field(default=128, metadata={"help": "Max decoding length."})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})