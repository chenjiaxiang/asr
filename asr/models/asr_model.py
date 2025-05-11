from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch.optim import ASGD, SGD, Adadelta, Adagrad, Adam, Adamax, AdamW

from asr.criterion import CRITERION_REGISTRY
from asr.metrics import CharacterErrorRate, WordErrorRate
from asr.optim import AdamP, Novograd, RAdam
from asr.optim.scheduler import SCHEDULER_REGISTRY
from asr.tokenizers.tokenizer import Tokenizer

# pause here