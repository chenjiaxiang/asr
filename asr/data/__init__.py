import importlib
import os

AUDIO_FEATURE_TRANSFORM_REGISTRY = dict()
AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY = dict()


def register_audio_feature_transform(name: str, dataclass=None):
    def register_audio_feature_transform_cls(cls):
        if name in AUDIO_FEATURE_TRANSFORM_REGISTRY:
            raise ValueError(f"Cannot register duplicate audio ({name})")

        AUDIO_FEATURE_TRANSFORM_REGISTRY[name] = cls

        cls.__dataclass__ = dataclass
        if dataclass is not None:
            if name in AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY:
                raise ValueError(f"Cannot register duplicate dataclass ({name})")
            AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY[name] = dataclass

        return cls

    return register_audio_feature_transform_cls

data_dir = os.path.dirname(__file__)
for file in os.listdir(os.path.join(data_dir, "audio")):
    if os.path.isdir(os.path.join(data_dir, "audio", file)) and not file.startswith("__"):
        path = os.path.join(data_dir, "audio", file)
        for module_file in os.listdir(path):
            path = os.path.join(path, module_file)
            if module_file.endswith(".py"):
                module_name = module_file[: module_file.find(".py")] if module_file.endswith(".py") else module_file
                module = importlib.import_module(f"asr.data.audio.{file}.{module_name}")