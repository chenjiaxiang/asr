from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.optim import Optimizer

from asr.dataclass.configurations import LearningRateSchedulerConfigs
from asr.optim.scheduler import register_scheduler
from asr.optim.scheduler.lr_scheduler import LearningRateScheduler


@dataclass
class WarmupLRScheduleConfigs(LearningRateSchedulerConfigs):
    scheduler_name: str = field(default="warmup", metadata={"help": "Name of learning rate scheduler."})
    peak_lr: float = field(default=1e-04, metadata={"help": "Maximum learning rate."})
    init_lr: float = field(default=1e-7, metadata={"help": "Initial learning rate."})
    warmup_steps: int = field(
        default=4000, metadata={"help": "Warmup the learning rate linearly for the first N updates."}
    )
    total_steps: int = field(default=200000, metadata={"help": "Total training steps."})


@register_scheduler("warmup", dataclass=WarmupLRScheduleConfigs)
class WarmupLRScheduler(LearningRateScheduler):
    def __init__(
            self, 
            optimizer: Optimizer,
            configs: DictConfig,
    ) -> None:
        super(WarmupLRScheduler, self).__init__(optimizer, configs.lr_scheduler.init_lr)
        if configs.lr_scheduler.warmup_steps != 0:
            warmup_rate = configs.lr_scheduler.peak_lr - configs.lr_scheduler.init_lr
            self.warmup_rate = warmup_rate / configs.lr_scheduler.warmup_steps
        else:
            self.warmup_rate = 0
        self.update_step = 1
        self.lr = configs.lr_scheduler.init_lr
        self.warmup_steps = configs.lr_scheduler.warmup_steps

    def step(self, val_loss: Optional[torch.FloatTensor] = None) -> float:
        if self.update_step  < self.warmup_steps:
            lr = self.init_lr + self.warmup_rate * self.update_step
            self.set_lr(self.optimizer, lr)
            self.lr = lr
        self.update_step += 1

        return self.lr
            