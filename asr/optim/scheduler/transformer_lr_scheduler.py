import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.optim import Optimizer

from asr.dataclass.configurations import LearningRateSchedulerConfigs
from asr.optim.scheduler import register_scheduler
from asr.optim.scheduler.lr_scheduler import LearningRateScheduler

@dataclass
class TransformerLRSchedulerConfigs(LearningRateSchedulerConfigs):
    scheduler_name: str = field(default="transformer", metadata={"help": "Name of learning rate scheduler."})
    peak_lr: float = field(default=1e-04, metadata={"help": "Maximum learning rete."})
    final_lr: float = field(default=1e-07, metadata={"help": "Final learning rate."})
    final_lr_scale: float = field(default=0.05, metadata={"help": "Final learning rate scale."})
    warmup_steps: int = field(
        default=10000, metadata={"help": "Warmup the learning rate linearly for the first N updates."}
    )
    decay_streps: int = field(default=150000, metadata={"help": "Steps in decay stages."})


@register_scheduler("transformer", dataclass=TransformerLRSchedulerConfigs)
class TransformerLRScheduler(LearningRateScheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            configs: DictConfig,
    ) -> None:
        assert isinstance(configs.lr_scheduler.warmup_steps, int), "warmup_steps should be integer type."
        assert isinstance(configs.lr_scheduler.decay_steps, int), "total_steps should be integer type."

        super(TransformerLRScheduler, self).__init__(optimizer, 0.0)
        self.peak_lr = configs.lr_scheduler.peak_lr
        self.final_lr = configs.lr_scheduler.final_lr
        self.warmup_steps = configs.lr_scheduler.warmup_steps
        self.decay_steps = configs.lr_scheduler.decay_steps

        self.warmup_rate = self.peak_lr / self.warmup_steps
        self.decay_factor = -math.log(configs.lr_scheduler.final_lr_scale) / self.decay_steps

        self.lr = self.init_lr
        self.update_step = 0

    def _decide_stage(self) -> Tuple[int, Optional[int]]:
        if self.update_step < self.warmup_steps:
            return 0, self.update_step
        
        if self.warmup_steps <= self.update_step < self.warmup_steps + self.decay_steps:
            return 1, self.update_step - self.warmup_steps

        return 2, None

    def step(self, val_loss: Optional[torch.FloatTensor] = None) -> float:
        self.update_step += 1
        stage, steps_in_the_stage = self._decide_stage()

        if stage == 0:
            self.lr = self.update_step * self.warmup_rate
        elif stage == 1:
            self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_the_stage)
        elif stage == 2:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage.")

        self.set_lr(self.optimizer, self.lr)

        return self.lr