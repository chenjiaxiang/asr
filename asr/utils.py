import importlib
import logging
import platform
from collections import OrderedDict
from typing import Iterable, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor


PYTORCH_IMPORT_ERROR = """
asr requires the PyTorch library but it was not found in your environment. Checkout the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
"""

TORCHAUDIO_IMPORT_ERROR = """
asr requires the torchaudio library but it was not found in your environment. You can install it with pip:
`pip install torchaudio`
"""

PYTORCH_LIGHTNING_IMPORT_ERROR = """
asr requires the pytorch-lightning library but it was not found in your environment. You can install it with pip:
`pip install pytorch-lightning`
"""

HYDRA_IMPORT_ERROR = """
asr requires the hydra library but it was not found in your environment. You can install it with pip:
`pip install hydra-core`
"""

LIBROSA_IMPORT_ERROR = """
asr requires the librosa library but it was not found in your environment. You can install it with pip:
`pip install librosa`
"""

SENTENCEPIECE_IMPORT_ERROR = """
asr requires the SentencePiece library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/google/sentencepiece#installation and follow the ones
that match your environment.
"""

WARPRNNT_IMPORT_ERROR = """
asr requires the warp-rnnt library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/1ytic/warp-rnnt and follow the ones that match your environment.
"""

CTCDECODE_IMPORT_ERROR = """
asr requires the ctcdecode library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/parlance/ctcdecode and follow the ones that match your environment.
"""

try:
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

def get_class_name(obj):
    return obj.__class__.__name__


def parse_configs(configs: DictConfig) -> Tuple[Union[TensorBoardLogger, bool], int]:
    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(configs))
    num_devices = _check_environment(configs.trainer.use_cuda, logger)

    if configs.trainer.logger == "tensorboard":
        logger = TensorBoardLogger("logs/")
    elif configs.trainer.logger == "wandb":
        logger = WandbLogger(
            project=f"{configs.model.model_name}-{configs.dataset.dataset}",
            name=f"{configs.model.model_name}-{configs.dataset.dataset}",
            job_type="train", # TODO maybe wrong
        )
    else:
        logger = True
    
    return logger, num_devices


def _check_environment(use_cuda: bool, logger) -> int:
    check_backends()

    cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    logger.info(f"Operating System: {platform.system()} {platform.release()}")
    logger.info(f"Processor: {platform.processor()}")

    num_devices = torch.cuda.device_count()

    if str(device) == "cuda":
        for idx in range(torch.cuda.device_count()):
            logger.info(f"device : {torch.cuda.get_device_name(idx)}")
        logger.info(f"CUDA is available : {torch.cuda.is_available()}")
        logger.info(f"CUDA version : {torch.version.cuda}")
        logger.info(f"PyTorch version : {torch.__version__}")
    
    else:
        logger.info(f"CUDA is available : {torch.cuda.is_available()}")
        logger.info(f"PyTorch version : {torch.__version__}")

    return num_devices


def is_pytorch_available():
    return importlib.util.find_spec("torch") is not None


def is_pytorch_lightning_available():
    return importlib.util.find_spec("pytorch_lightning") is not None


def is_hydra_available():
    return importlib.util.find_spec("hydra") is not None


def is_librosa_available():
    return importlib.util.find_spec("librosa") is not None


def is_apex_available():
    return importlib.util.find_spec("apex") is not None


def is_sentencepiece_available():
    return importlib.util.find_spec("sentencepiece") is not None


def is_torchaudio_available():
    return importlib.util.find_spec("torchaudio") is not None

BACKENDS_MAPPING = OrderedDict(
    [
        ("torch", (is_pytorch_available, PYTORCH_IMPORT_ERROR)),
        ("sentencepiece", (is_sentencepiece_available, SENTENCEPIECE_IMPORT_ERROR)),
        ("pytorch_lighting", (is_pytorch_lightning_available, PYTORCH_LIGHTNING_IMPORT_ERROR)),
        ("hydra", (is_hydra_available, HYDRA_IMPORT_ERROR)),
        ("librosa", (is_librosa_available, LIBROSA_IMPORT_ERROR)),
        ("torchaudio", (is_torchaudio_available, TORCHAUDIO_IMPORT_ERROR)),
    ]
)


def check_backends():
    backends = BACKENDS_MAPPING.keys()

    if not all(BACKENDS_MAPPING[backend][0] for backend in backends):
        raise ImportError("".join([BACKENDS_MAPPING[backend][1] for backend in backends]))

def get_pl_trainer()
# pause here