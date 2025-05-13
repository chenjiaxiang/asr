import importlib
import os

DATA_MODULE_REGISTRY = dict()

def register_data_module(name: str):
    def register_data_module_cls(cls):
        if name in DATA_MODULE_REGISTRY:
            raise ValueError(f"Cannot register duplicate data module ({name})")
        DATA_MODULE_REGISTRY[name] = cls
        return cls

    return register_data_module_cls

data_module_dir = os.path.dirname(__file__)
for file in os.listdir(data_module_dir):
    if os.path.isdir(os.path.join(data_module_dir, file)) and file != "__pycache__":
        for subfile in os.listdir(os.path.join(data_module_dir, file)):
            path = os.path.join(data_module_dir, file, subfile)
            if subfile.endswith(".py"):
                data_module_name = subfile[: subfile.find(".py")] if subfile.endswith(".py") else subfile
                module = importlib.import_module(f"asr.datasets.{file}.{data_module_name}")