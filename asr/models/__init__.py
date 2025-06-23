import importlib
import os

from .asr_model import ASRModel 
# from .asr_ctc_model import ASRCTCModel
from .asr_encoder_decoder import ASREncoderDecoderModel

MODEL_REGISTRY = dict()
MODEL_DATACLASS_REGISTRY = dict()

def register_model(name: str, dataclass=None):
    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Cannot register duplicate model ({name})")
        if not issubclass(cls, ASRModel):
            raise ValueError(f"Model ({name}: {cls.__name__}) must extend ASRModel")

        MODEL_REGISTRY[name] = cls

        cls.__dataclass = dataclass
        if dataclass is not None:
            if name in MODEL_DATACLASS_REGISTRY:
                raise ValueError(f"Cannot register duplicate model ({name})")
            MODEL_DATACLASS_REGISTRY[name] = dataclass
        
        return cls
    
    return register_model_cls

# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    if os.path.isdir(os.path.join(models_dir, file)) and not file.startswith("__"):
        for subfile in os.listdir(os.path.join(models_dir, file)):
            path = os.path.join(models_dir, file, subfile)
            if subfile.endswith(".py"):
                python_file = subfile[: subfile.find(".py")] if subfile.endswith(".py") else subfile
                module = importlib.import_module(f"asr.models.{file}.{python_file}")
        continue

from .conformer import (
    ConformerConfigs,
    ConformerLSTMConfigs,
    ConformerLSTMModel,
)

__all__ = [
    "ASRModel",
    "ASREncoderDecoderModel",
    # "ASRCTCModel",
    "ConformerConfigs",
    "ConformerLSTMConfigs",
    "ConformerLSTMModel",
]
    