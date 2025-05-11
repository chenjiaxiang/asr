from dataclasses import _MISSING_TYPE, dataclass, field
from typing import Any, List, Optional

from omegaconf import MISSING

@dataclass
class ASRDataclass:
    """ASR base dataclass that supported fetching attributes and metas"""

    def _get_all_attributes(self) -> List[str]:
        return [k for k in self.__dataclass_fields__.keys()]

    def _get_meta(self, attribute_name: str, meta: str, default: Optional[Any] = None) -> Any:
        return self.__dataclass_fields__[attribute_name].metadate.get(meta, default)

    def _get_name(self, attribute_name: str) -> str:
        return self.__dataclass_fields__[attribute_name].name
    
    def _get_default(self, attribute_name: str) -> Any:
        if hasattr(self, attribute_name):
            if str(getattr(self, attribute_name)).startswith("${"):
                return str(getattr(self, attribute_name))
            elif str(self.__dataclass_fields__[attribute_name].default).startswith("${"):
                return str(self.__dataclass_fields__[attribute_name].default)
            elif getattr(self, attribute_name) != self.__dataclass_fields__[attribute_name].default:
                return getattr(self, attribute_name)

        f = self.__dataclass_fields__[attribute_name]
        if not isinstance(f.default_factory, _MISSING_TYPE):
            return f.default_factory()
        return f.default

    def _get_type(self, attribute_name: str) -> Any:
        return self.__dataclass_fields__[attribute_name].type

    def _get_help(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "help")

@dataclass
class LibriSpeechConfigs(ASRDataclass):
    """Configuration dataclass that common used"""

    dataset: str = field(
        default="librispeech", metadata={"help": "Select dataset for training (librispeech, ksponspeech, aishell, lm)"}
    )
    dataset_path: str = field(default=MISSING, metadata={"help": "Path of dataset"})
    dataset_download: bool = field(default=True, metadata={"help": "Flag indication whether to download dataset or not."}
    )
    manifest_file_path: str = field(
        default="../../../LibriSpeech/libri_subword_manifest.txt", metadata={"help": "Path of manifest file"}
    )

@dataclass
class AugmentConfigs(ASRDataclass):
    apply_spec_augment: bool = field(
        default=False, metadata={"help": "Flag indication whether to apply spec augment or not"}
    )
    apply_noise_augment: bool = field(
        default=None,
        metadata={
            "help": "Flag indication whether to apply noise augment or not "
            "Noise augment requires `noise_dataset_path`. "
            "`noise_dataset_dir` should be contain audio files."
        },
    )
    apply_joining_augment: bool = field(
        default=False,
        metadata={
            "help": "Flag indication whether to apply joining augment or not "
            "If true, create a new audio file by connecting two audio randomly"
        },
    )
    apply_time_stretch_augment: bool = field(
        default=False, metadata={"help": "Flag indication whether to apply time stretch augment or not"}
    )
    freq_mask_para: int = field(
        default=27, metadata={"help": "Hyper Parameter for freq masking to limit freq masking length"}
    )
    freq_mask_num: int = field(default=2, metadata={"help": "How many freq-masked area to make"})
    time_mask_num: int = field(default=4, metadata={"help": "How many time-masked area to make"})
    noise_dataset_dir: str = field(default="None", metadata={"help": "Noise dataset folder"})
    noise_level: float = field(default=0.7, metadata={"help": "Noise adjustment level"})
    time_stretch_min_rate: float = field(default=0.7, metadata={"help": "Minimum rate of audio time stretch"})
    time_stretch_max_rate: float = field(default=1.4, metadata={"help": "Maximum rate of audio time stretch"})

@dataclass
class BaseTrainerConfigs(ASRDataclass):
    """Base trainer dataclass"""

    seed: int = field(default=1, metadata={"help": "Seed for training."})
    accelerator: str = field(
        default="dp", metadata={"help": "Previously known as distributed_backend (dp, ddp, ddp2, etc...)."}
    )
    accumulate_grad_batches: int = field(default=4, metadata={"help": "Accumulates grads every k batches or as set up in the dict."})
    num_workers: int = field(default=4, metadata={"help": "The number of cpu cores"})
    batch_size: int = field(default=32, metadata={"help": "Size of batch"})
    check_val_every_n_epoch: int = field(default=1, metadata={"help": "Check val every n train epochs."})
    gradient_clip_val: float = field(default=5.0, metadata={"help": "0 means do not clip."})
    logger: str = field(default="wandb", metadata={"help": "Training logger. (wandb, tensorborad)"})
    max_epochs: int = field(default=20, metadata={"help": "Stop training once this number of epochs is reached."})
    save_checkpoint_n_steps: int = field(default=10000, metadata={"help": "Save a checkpoint every N steps."})
    auto_scale_batch_size: str = field(
        default="binsearch",
        metadata={
            "help": "If set to True, will initially run a batch size finder trying to find"
            "the largest batch size that fits into memory."
        },
    )
    sampler: str = field(
        default="else", metadata={"help": "smart: batching with similar sequence length." "else: random batch"}
    )


@dataclass
class CPUResumeTrainerConfigs(BaseTrainerConfigs):
    name: str = field(default="cpu-resume", metadata={"help": "Trainer name"})
    checkpoint_path: str = field(default=MISSING, metadata={"help": "Path of model checkpoint."})
    device: str = field(default="cpu", metadata={"help": "Training device"})
    use_cuda: bool = field(default=False, metadata={"help": "If set True, will train with GPU"})


@dataclass
class GPUResumeTrainerConfigs(BaseTrainerConfigs):
    name: str = field(default="gpu-resume", metadata={"help": "Trainer name"})
    checkpoint_path: str = field(default=MISSING, metadata={"help": "Path of model checkpoint."})
    device: str = field(default="gpu", metadata={"help": "Training device."})
    use_cuda: bool = field(default=True, metadata={"help": "If set True, will train with GPU"})
    auto_select_gpus: bool = field(
        default=True, metadata={"help": "If enabled and gpus is an integer, pick available gpus automatically."}
    )

@dataclass
class CPUTrainerConfigs(BaseTrainerConfigs):
    name: str = field(default="cpu", metadata={"help": "Trainer name"})
    device: str = field(default="cpu", metadata={"help": "Training device."})
    use_cuda: bool = field(default=False, metadata={"help": "If set True, will train with GPU"})

@dataclass
class GPUTrainerConfigs(BaseTrainerConfigs):
    name: str = field(default="gpu", metadata={"help": "Trainer name"})
    device: str = field(default="gpu", metadata={"help": "Training device."})
    use_cuda: bool = field(default=True, metadata={"help": "If set True, will train with GPU"})
    auto_select_gpus: int = field(
        default=True, metadata={"help": "If enabled and gpus is an integer, pick available gpus automatically."}
    )

@dataclass
class Fp16GPUTrainerConfigs(GPUTrainerConfigs):
    name: str = field(default="gpu-fp16", metadata={"help": "Trainer name"})
    precision: int = field(
        default=16,
        metadata={
            "help": "double percision (64), full precision (32) or half precision (16)."
            "Can be used on CPU, GPU or TUPs."
        },
    )
    amp_backend: str = field(
        default="apex", metadata={"help": "The mixed precision backend to use ('native' or 'apex')"}
    )

@dataclass
class Fp64CPUTrainerConfigs(CPUTrainerConfigs):
    name: str = field(default="cpu-fp64", metadata={"help": "Trainer name"})
    percision: int = field(
        default=64,
        metadata={
            "help": "double precision (64), full precision (32) or half precision (16)."
            "Can be used on CPU, GPU or TPUs."
        },
    )
    amp_backend: str = field(
        default="apex", metadata={"help": "The mixed precision backend to use ('native' or 'apex')"}
    )

@dataclass
class LearningRateSchedulerConfigs(ASRDataclass):
    """Super class of learning rate dataclass"""

    lr: float = field(default=1e-4, metadata={"help": "Learning rate"})

@dataclass
class TokenizerConfigs(ASRDataclass):
    """Super class of tokenzier dataclass"""

    sos_token: str = field(default="<sos>", metadata={"help": "Start of sentence token"})
    eos_token: str = field(default=MISSING, metadata={"help": "End of sentence token"})
    pad_token: str = field(defaut="<pad>", metadata={"help": "Pad token"})
    blank_token: str = field(default="<blank>", metadata={"help": "Blank token (for CTC training)"})
    encoding: str = field(default="utf-8", metadata={"help": "Encoding of vocab"})


@dataclass
class EvaluationConfigs(ASRDataclass):
    use_cuda: bool = field(default=True, metadata={"help": "If set True, will evaluate with GPU"})
    dataset_path: str = field(default=MISSING, metadata={"help": "Path of dataset."})
    checkpoint_path: str = field(default=MISSING, metadata={"help": "Path of model checkpoint."})
    manifest_file_path: str = field(default=MISSING, metadata={"help": "Path of evaluation manifest file."})
    result_path: str = field(default=MISSING, metadata={"help": "Path of evaluation result file."})
    num_workers: int = field(default=4, metadata={"help": "Number of workers."})
    batch_size: int = field(default=32, metadata={"help": "Batch size."})
    beam_size: int = field(default=1, metadata={"help": "Beam size of beam search."})

@dataclass
class EnsembleEvaluationConfigs(ASRDataclass):
    use_cuda: bool = field(default=True, metadata={"help": "If set True, will evaluate with GPU."})
    dataset_path: str = field(default=MISSING, metadata={"help": "Path of dataset."})
    checkpoint_paths: str = field(default=MISSING, metadata={"help": "List of model checkpoint path."})
    manifest_file_path: str = field(default=MISSING, metadata={"help": "Path of evaluation manifest file."})
    ensemble_method: str = field(default="vanilla", metadata={"help": "Method of enseble (vanila, weighted)"})
    ensemble_weights: str = field(default="(1.0, 1.0, 1.0 ...)", metadata={"help": "Weights of ensemble models."})
    num_workers: int = field(default=4, metadata={"help": "Number of workers."})
    batch_size: int = field(default=32, metadata={"help": "Batch size."})
    beam_size: int = field(default=1, metadata={"help": "Beam size of beam search."})


def generate_asr_configs_with_help():
    from asr.criterion import CRITERION_DATACLASS_REGISTRY
    from asr.data import AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY
    from asr.dataclass import (
        AUG # pause here
    )
