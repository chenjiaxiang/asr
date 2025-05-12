from dataclasses import dataclass, field
from typing import Optional

from omegaconf import DictConfig
from torch.optim import Optimizer

from asr.dataclass.configurations import LearningRateSchedulerConfigs
from asr.optim.scheduler import register_scheduler
from asr.optim.scheduler.lr_scheduler import LearningRateScheduler

@dataclass
class ReduceLROnPlateauConfigs(LearningRateSchedulerConfigs):
    scheduler_name: str = field(default="reduce_lr_on_plateau", metadata={"help": "Name of learning rate scheduler."})
    lr_patience: int = field(
        default=1, metadata={"help": "Number of epochs with no improvement after which learning rate will be reduced."}
    )
    lr_factor: float = field(
        default=0.3, metadata={"help": "Factor by which the learning rate will be reduced. new_lr = lr * factor."}
    )

@register_scheduler("reduce_lr_on_plateau", dataclass=ReduceLROnPlateauConfigs)
class ReduceLROnPlateauScheduler(LearningRateScheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            configs: DictConfig,
    ) -> None:
        super(ReduceLROnPlateauScheduler, self).__init__(optimizer, configs.lr_scheduler.lr)
        self.lr = configs.lr_scheduler.lr
        self.lr_patience = configs.lr_scheduler.lr_patience
        self.lr_factor = configs.lr_scheduler.lr_factor
        self.val_loss = 100.0
        self.count = 0
        
    def step(self, val_loss: Optional[float] = None) -> float:
        if val_loss is not None:
            if self.val_loss < val_loss:
                self.count += 1
                self.val_loss = val_loss
            else:
                self.count = 0
                self.val_loss = val_loss
            
            if self.lr_patience == self.count:
                self.count = 0
                self.lr *= self.lr_factor
                self.set_lr(self.optimizer, self.lr)
            
        return self.lr