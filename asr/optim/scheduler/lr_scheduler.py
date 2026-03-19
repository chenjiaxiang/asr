from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LearningRateScheduler(_LRScheduler):
    r"""Abstract base class for learning rate schedulers.

    Extends ``torch.optim.lr_scheduler._LRScheduler`` with a simple
    interface for setting and getting the current learning rate. Subclasses
    must implement :meth:`step`.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        init_lr (float): Initial learning rate.

    Examples::

        >>> class MyScheduler(LearningRateScheduler):
        ...     def step(self, val_loss=None):
        ...         self.set_lr(self.optimizer, self.init_lr)
        ...         return self.init_lr
    """
    def __init__(self, optimizer: Optimizer, init_lr: float) -> None:
        self.optimizer = optimizer
        self.init_lr = init_lr

    def step(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def set_lr(optimizer: Optimizer, lr: float) -> None:
        for g in optimizer.param_groups:
            g["lr"] = lr

    def get_lr(self) -> float:
        for g in self.optimizer.param_groups:
            return g["lr"]
