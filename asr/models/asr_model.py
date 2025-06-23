from typing import Dict, Tuple

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

class ASRModel(pl.LightningModule):
    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(ASRModel, self).__init__()
        self.configs = configs
        self.num_classes = len(tokenizer)
        self.tokenizer = tokenizer
        self.current_val_loss = 100.0
        self.wer_metric = WordErrorRate(tokenizer)
        self.cer_metric = CharacterErrorRate(tokenizer)
        if hasattr(configs, "trainer"):
            self.gradient_clip_val = configs.trainer.gradient_clip_val
        if hasattr(configs, "criterion"):
            self.criterion = self.configure_criterion(configs.criterion.criterion_name)
    
    def set_beam_decoder(self, beam_size: int = 3):
        raise NotImplementedError

    def info(self, dictionary: dict) -> None:
        for key, value in dictionary.items():
            self.log(key, value, prog_bar=True)

    def forward(self, inputs: torch.LongTensor, input_lengths: torch.LongTensor) -> Dict[str, Tensor]:
        raise NotImplementedError

    def training_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        raise NotImplementedError

    def validation_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        raise NotImplementedError

    def test_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        raise NotImplementedError

    def configure_optimizers(self):
        SUPPORTED_OPTIMIZERS = {
            "adam": Adam,
            "adamp": AdamP,
            "radam": RAdam,
            "adagrad": Adagrad,
            "adadelta": Adadelta,
            "adamax": Adamax,
            "adamw": AdamW,
            "sgd": SGD,
            "asgd": ASGD,
            "novograd": Novograd,
        }

        assert self.configs.model.optimizer in SUPPORTED_OPTIMIZERS.keys(), (
            f"Unsupported Opitmizer: {self.configs.model.optimizer}\n"
            f"Supported Optimizers: {SUPPORTED_OPTIMIZERS.keys()}"
        )

        self.optimizer = SUPPORTED_OPTIMIZERS[self.configs.model.optimizer](
            self.parameters(),
            lr=self.configs.lr_scheduler.lr
        )
        scheduler = SCHEDULER_REGISTRY[self.configs.lr_scheduler.scheduler_name](self.optimizer, self.configs)

        if self.configs.lr_scheduler.scheduler_name == "reducr_lr_on_plateau":
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            }
        elif self.configs.lr_scheduler.scheduler_name == "warmup_reduce_lr_on_plateau":
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "step",
            }
        else:
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
            }

        return [self.optimizer], [lr_scheduler]

    def configure_criterion(self, criterion_name: str) -> nn.Module:
        if criterion_name in ("joint_ctc_cross_entropy", "label_smoothed_cross_entropy"):
            return CRITERION_REGISTRY[criterion_name](
                configs=self.configs,
                num_classes=self.num_classes,
                tokenizer=self.tokenizer,
            )
        else:
            return CRITERION_REGISTRY[criterion_name](
                configs=self.configs,
                tokenizer=self.tokenizer,
            )

    def get_lr(self) -> float:
        for g in self.optimizer.param_groups:
            return g["lr"]

    def set_lr(self, lr: float) -> None:
        for g in self.optimizer.param_groups:
            g["lr"] = lr
