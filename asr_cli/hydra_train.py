import os
import hydra
import sentencepiece
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_info

from asr.dataclass.initialize import hy