from torch.optim.lr_scheduler import _LRScheduler

class LearningRateScheduler(_LRScheduler):
    def __init__(self, optimizer, init_lr):
        self.optimizer = optimizer
        self.init_lr = init_lr # TODO maybe wrong
    
    def step(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g["lr"] = lr

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g["lr"]   # TODO maybe wrong