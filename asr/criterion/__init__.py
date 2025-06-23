import importlib
import os

CRITERION_REGISTRY = dict()
CRITERION_DATACLASS_REGISTRY = dict()

def register_criterion(name: str, dataclass=None):
    def register_criterion_cls(cls):
        if name in CRITERION_REGISTRY:
            raise ValueError(f"Cannot register duplicate criterion ({name})")
        
        CRITERION_REGISTRY[name] = cls

        cls.__dataclass__ = dataclass
        if dataclass is not None:
            if name in CRITERION_DATACLASS_REGISTRY:
                raise ValueError(f"Cannot register duplicate criterion ({name})")
            CRITERION_DATACLASS_REGISTRY[name] = dataclass

        return cls

    return register_criterion_cls

criterion_dir = os.path.dirname(__file__)
for file in os.listdir(criterion_dir):
    if os.path.isdir(os.path.join(criterion_dir, file)) and not file.startswith("__"):
        for subfile in os.listdir(os.path.join(criterion_dir, file)):
            path = os.path.join(criterion_dir, file, subfile)
            if subfile.endswith(".py"):
                python_file = subfile[: subfile.find(".py")] if subfile.endswith(".py") else subfile # TODO fix
                module =  importlib.import_module(f"asr.criterion.{file}.{python_file}")
            continue

    path = os.path.join(criterion_dir, file)
    if file.endswith(".py"):
        criterion_name = file[: file.find(".py")] if file.endswith(".py") else file # TODO fix
        module = importlib.import_module(f"asr.criterion.{criterion_name}")

from .cross_entropy.configuration import CrossEntropyLossConfigs
from .cross_entropy.cross_entropy import CrossEntropyLoss
from .ctc.configuration import CTCLossConfigs
from .ctc.ctc import CTCLoss

__all__ = [
    "CrossEntropyLossConfigs",
    "CTCLossConfigs",
    "CrossEntropyLoss",
    "CTCLoss",
]