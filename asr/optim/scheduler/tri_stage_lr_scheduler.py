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
class TriStageLRSchedulerConfigs(LearningRateSchedulerConfigs):
    scheduler_name: str = field(default="tri_stage", metadata={"help": "Name of learning rate scheduler."})
    init_lr: float = field(default=1e-7, metadata={"help": "Initial learning rate."})
    init_lr_scale: float = field(default=0.01, metadata={"help": "Initial learning rate scale."})
    final_lr_scale: float = field(default=0.01, metadata={"help": "Final learning rate scale."})
    phase_ratio: str = field(
        default="(0.1, 0.4, 0.5)",
        metadata={"help": "Automatically set warmup/hold/decay steps to the ratio "
                  "specified here from max_updates. the ratios must add up to 1.0"
        },
    )
    total_steps: int = field(default=400000, metadata={"help": "Total training steps."})


@register_scheduler("tri_stage", dataclass=TriStageLRSchedulerConfigs)
class TriStageLRScheduler(LearningRateScheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            configs: DictConfig,
    ) -> None:
        super(TriStageLRScheduler, self).__init__(optimizer, configs.lr_scheduler.init_lr)

        self.phase_ratio = eval(configs.lr_scheduler.phase_ratio)

        self.warmup_steps = int(configs.lr_scheduler.total_steps * self.phase_ratio[0])
        self.hold_steps = int(configs.lr_scheduler.total_step * self.phase_ratio[1])
        self.decay_steps = int(configs.lr_scheduler.total_steps * self.phase_ratio[2])

        self.peak_lr = configs.lr_scheduler.lr
        self.init_lr = configs.lr_scheduler.init_lr_scale * configs.lr_scheduler.lr
        self.final_lr = configs.lr_scheduler.final_lr_scale * configs.lr_schedulder.lr

        self.warmup_rate = (self.peak_lr - self.init_lr) / self.warmup_steps if self.warmup_steps != 0 else 0
        self.decay_factor = -math.log(configs.lr_scheduler.final_lr_scale) / self.decay_steps
        self.update_step = 0
        self.lr = self.init_lr
    
    def _decide_stage(self) -> Tuple[int, Optional[int]]:
        if self.update_step < self.warmup_steps:
            return 0, self.update_step
        
        offset = self.warmup_steps

        if self.update_step < offset + self.hold_steps:
            return 1, self.update_step - offset

        offset += self.hold_steps

        if self.update_step <= offset + self.decay_steps:
            return 2, self.update_step - offset

        offset += self.decay_steps

        return 3, self.update_step - offset

    def step(self, val_loss: Optional[torch.FloatTensor] = None) -> float:
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.lr = self.init_lr + self.warmup_rate * steps_in_stage
        elif stage == 1:
            self.lr = self.peak_lr
        elif stage == 2:
            self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        elif stage == 3:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage.")
        
        self.set_lr(self.optimizer, self.lr)
        self.update_step += 1

        return self.lr