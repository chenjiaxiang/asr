from hydra.core.config_store import ConfigStore


def hydra_train_init() -> None:
    r"""initialize ConfigStore for hydra-train"""
    from asr.criterion import CRITERION_DATACLASS_REGISTRY
    from asr.data import AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY
    from asr.dataclass import (
        AUGMENT_DATACLASS_REGISTRY,
        DATASET_DATACLASS_REGISTRY,
        ASR_TRAIN_CONFIGS,
        TRAINER_DATACLASS_REGISTRY,
    )
    from asr.models import MODEL_DATACLASS_REGISTRY
    from asr.optim.scheduler import SCHEDULER_DATACLASS_REGISTRY
    from asr.tokenizers import TOKENIZER_DATACLASS_REGISTRY

    registries = {
        "audio": AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY,
        "augment": AUGMENT_DATACLASS_REGISTRY,
        "dataset": DATASET_DATACLASS_REGISTRY,
        "trainer": TRAINER_DATACLASS_REGISTRY,
        "model": MODEL_DATACLASS_REGISTRY,
        "criterion": CRITERION_DATACLASS_REGISTRY,
        "lr_scheduler": SCHEDULER_DATACLASS_REGISTRY,
        "tokenizer": TOKENIZER_DATACLASS_REGISTRY,
    }

    cs = ConfigStore.instance()

    for group in ASR_TRAIN_CONFIGS:
        dataclass_registry = registries[group]

        for k, v in dataclass_registry.items():
            cs.store(group=group, name=k, node=v, provider="asr")


def hydra_eval_init() -> None:
    from asr.data import AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY
    from asr.dataclass import EVAL_DATACLASS_REGISTRY
    from asr.models import MODEL_DATACLASS_REGISTRY
    from asr.tokenizers import TOKENIZER_DATACLASS_REGISTRY

    registries = {
        "audio": AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY,
        "eval": EVAL_DATACLASS_REGISTRY,
        "model": MODEL_DATACLASS_REGISTRY,
        "tokenizer": TOKENIZER_DATACLASS_REGISTRY,
    }

    cs = ConfigStore.instance()

    for group in registries.keys():
        dataclass_registry = registries[group]

        for k, v in dataclass_registry.items():
            cs.store(group=group, name=k, node=v, provider="asr")