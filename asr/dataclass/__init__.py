from .configurations import (
    AugmentConfigs,
    CPUResumeTrainerConfigs,
    CPUTrainerConfigs,
    EnsembleEvaluationConfigs,
    EvaluationConfigs,
    Fp16GPUTrainerConfigs,
    Fp64CPUTrainerConfigs,
    GPUResumeTrainerConfigs,
    GPUTrainerConfigs,
    LibriSpeechConfigs,
)

ASR_TRAIN_CONFIGS = [
    "audio",
    "augment",
    "dataset",
    "model",
    "criterion",
    "lr_scheduler",
    "trainer",
    "tokenizer",
]

DATASET_DATACLASS_REGISTRY = {
    "librispeech": LibriSpeechConfigs,
}

TRAINER_DATACLASS_REGISTRY = {
    "cpu": CPUTrainerConfigs,
    "gpu": GPUTrainerConfigs,
    "gpu-fp16": Fp16GPUTrainerConfigs,
    "cpu-fp64": Fp64CPUTrainerConfigs,
    "cpu-resume": CPUResumeTrainerConfigs,
    "gpu-resume": GPUResumeTrainerConfigs,
}

AUGMENT_DATACLASS_REGISTRY = {
    "default": AugmentConfigs,
}

EVAL_DATACLASS_REGISTRY = {
    "default": EvaluationConfigs,
    "ensemble": EnsembleEvaluationConfigs,
}