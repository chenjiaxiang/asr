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

def get_class_name(obj):
    return obj.__class__.__name__