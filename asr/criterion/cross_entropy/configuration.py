from dataclasses import dataclass, field

from ...dataclass.configurations import ASRDataclass

@dataclass
class CrossEntropyLossConfigs(ASRDataclass):
    criterion_name: str = field(default="cross_entropy", metadata={"help": "Criterion name for training."})
    reduction: str = field(default="mean", metadata={"help": "Reduction method of criterion."})