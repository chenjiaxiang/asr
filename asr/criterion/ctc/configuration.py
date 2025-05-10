from dataclasses import dataclass, field

from ...dataclass.configurations import ASRDataclass


@dataclass
class CTCLossConfigs(ASRDataclass):
    criterion_name: str = field(default="ctc", metadata={"help": "Criterion name for training."})
    reduction: str = field(default="mean", metadata={"help": "Reduction method of criterion."})
    zero_infinity: bool = field(
        default=True, metadata={"help": "Whether to zero infinite losses and the associated gradients."}
    )