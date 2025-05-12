from dataclasses import dataclass, field
from typing import Optional, Tuple

from omegaconf import DictConfig
from torch.optim import Optimizer

from asr.dataclass.configurations import LearningRateSchedulerConfigs
from asr.optim.scheduler import register_scheduler
from asr.optim.scheduler.lr_scheduler import LearningRateScheduler
from asr.optim.scheduler.reduce_lr_on_plateau_scheduler import ReduceLROnPlateauConfigs
from asr.optim.scheduler.warmup_scheduler import WarmupLRScheduler


@dataclass
class WarmupReduceRLOnPlateauConfigs(LearningRateSchedulerConfigs):
    scheduler_name: str = field(
        default="warmup_reduce_lr_on_plateau", metadata={"help": "Name of learning rate scheduler."}
    )
    lr_patience: int = field(
        default=1, metadata={"help": "Number of epochs with no improvement after which learning rate will be reduced."}
    )
    lr_factor: float = field(
        default=0.3, metadata={"help": "Factorr by which the learning rate will be reduced. new_lr = lr * factor."}
    )
    peak_lr: float = field(default=1e-04, metadata={"help": "Maximum learning rate."})
    init_lr: float = field(default=1e-10, metadata={"help": "Initial learning rate."})
    warmup_steps: int = field(
        default=4000, metadata={"help": "Warmup the learning rate linearly for the first N updates."}
    )


@register_scheduler("warmup_reduce_lr_on_plateau", dataclass=WarmupReduceRLOnPlateauConfigs)
class WarmupReduceLROnPlateauScheduler(LearningRateScheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            configs: DictConfig,
    ) -> None:
        self.warmup_steps = configs.lr_scheduler.warmup_steps
        self.update_steps = 0
        self.warmup_rate = (
            (configs.lr_scheduler.peak_lr - configs.lr_scheduler.init_lr) / self.warmup_steps
            if self.warmup_steps != 0
            else 0
        )
        self.schedulers = [
            WarmupLRScheduler(
                optimizer,
                configs,
            ),
            ReduceLROnPlateauConfigs(
                optimizer,
                configs,
            ),
        ]

    def _decide_stage(self) -> Tuple[int, Optional[int]]:
        if self.update_steps < self.warmup_steps:
            return 0, self.update_steps
        else:
            return 1, None

    def step(self, val_loss: Optional[float] = None) -> float:
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.schedulers[0].step()
        elif stage == 1:
            self.schedulers[1].step(val_loss)

        self.update_steps += 1
        return self.get_lr()    # TODO maybe wrong, in the above two schedulers step(), where set lr