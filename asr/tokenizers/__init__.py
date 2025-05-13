import importlib
import os

TOKENIZER_REGISTRY = dict()
TOKENIZER_DATACLASS_REGISTRY = dict()

def register_tokenizer(name: str, dataclass=None):
    def register_tokenizer_cls(cls):
        if name in TOKENIZER_REGISTRY:
            raise ValueError(f"Cannot register duplicate tokenizer ({name})")
        
        TOKENIZER_REGISTRY[name] = cls

        cls.__dataclass__ = dataclass
        if dataclass is not None:
            if name in TOKENIZER_DATACLASS_REGISTRY:
                raise ValueError(f"Cannot register duplicate tokenizer ({name})")
            TOKENIZER_DATACLASS_REGISTRY[name] = dataclass

        return cls

    return register_tokenizer_cls


tokenizer_dir = os.path.dirname(__file__)
for file in os.listdir(tokenizer_dir):
    if os.path.isdir(os.path.join(tokenizer_dir, file)) and file != "__pycache__": # TODO differ from other
        for subfile in os.listdir(os.path.join(tokenizer_dir, file)):
            path = os.path.join(tokenizer_dir, file, subfile)
            if subfile.endswith(".py"):
                tokenizer_name = subfile[: subfile.find(".py")] if subfile.endswith(".py") else file
                module = importlib.import_module(f"asr.tokenizer.{file}.{tokenizer_name}")
        continue

    path = os.path.join(tokenizer_dir, file)   # TODO why?
    if file.endswith(".py"):
        vocab_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module(f"asr.tokenizer.{vocab_name}")

        