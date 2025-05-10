from dataclasses import dataclass, field

from ....dataclass.configurations import ASRDataclass


@dataclass
class MFCCConfigs(ASRDataclass):
    name: str = field(default="mfcc", metadata={"help": "Name of feature."})
    sample_rate: int = field(default=16000, metadata={"help": "Sampling rate of the audio."})
    frame_length: float = field(default=20.0, metadata={"help": "Frame length for spectrogram."})
    frame_shift: float = field(default=10.0, metadata={"help": "Length of hop between STFT."})
    del_silence: bool = field(
        default=False, metadata={"help": "Flag indication whether to apply delete silence or not."}
    )
    num_mels: int = field(default=80, metadata={"help": "The number of mfcc coefficients to retain."})
    apply_spec_augment: bool = field(
        default=True, metadata={"help": "Flag indication whether to apply spec augment or not."}
    )
    apply_noise_augment: bool = field(
        default=False, metadata={"help": "Flag indication whether to apply noise augment or not."}
    )
    apply_time_stretch_augment: bool = field(
        default=False, metadata={"help": "Flag indication whether to apply time stretch augment or not."}
    )
    apply_joining_augment: bool = field(
        default=False, metadata={"help": "Flag indication whether to apply audio joining augment or not."}
    )