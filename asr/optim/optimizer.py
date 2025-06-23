import torch

from asr.optim.scheduler.reduce_lr_on_plateau_scheduler import ReduceLROnPlateauScheduler
from asr.optim.scheduler.warmup_reduce_lr_on_plateau_scheduler import WarmupReduceLROnPlateauScheduler

class Optimizer(object):
    def __init__(self, optim, scheduler=None, scheduler_period=None, max_grad_norm=0):
        self.optimizer = optim
        self.scheduler = scheduler
        self.scheduler_period = scheduler_period
        self.max_grad_norm = max_grad_norm
        self.count = 0

    def step(self, model):
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if self.scheduler is not None:
            self.update()
            self.count += 1

            if self.scheduler_period == self.count:
                self.scheduler = None
                self.scheduler_period = 0
                self.count = 0

    def set_scheduler(self, scheduler, scheduler_period):
        self.scheduler = scheduler
        self.scheduler_period = scheduler_period
        self.count = 0

    def update(self, val_loss=None):
        if isinstance(self.scheduler, ReduceLROnPlateauScheduler) or isinstance(
            self.scheduler, WarmupReduceLROnPlateauScheduler
        ):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def ger_lr(self):
        for g in self.optimizer.param_groups:
            return g["lr"]

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g["lr"] = lr
