import numpy as np
from abc import ABC, abstractmethod

from ..core.register import SCHEDULER_REGISTER

class BaseScheduler(ABC):
    def __init__(self):
        super(BaseScheduler, self).__init__()

    @classmethod
    @abstractmethod
    def init_from_args(cls, args):
        raise NotImplementedError

    @abstractmethod
    def get_schedule(self):
        raise NotImplementedError

# copy-paste from https://github.com/facebookresearch/dino/blob/main/utils.py
@SCHEDULER_REGISTER.register('cosine')
class CosineScheduler(BaseScheduler):
    def __init__(self, base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
        super(CosineScheduler, self).__init__()
        self.schedule = self.init_schedule(base_value, final_value, epochs, niter_per_ep, warmup_epochs, start_warmup_value)

    @classmethod
    def init_from_args(cls, args):
        instance = cls(args.lr_start, args.lr_end, args.epochs, args.iters_per_epoch, args.lr_warmup, args.warmup_lr_start)
        return instance

    def init_schedule(self, base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
                warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == epochs * niter_per_ep
        return schedule
            
    def get_schedule(self):
        return self.schedule

@SCHEDULER_REGISTER.register('warmup')
class WarmupScheduler(BaseScheduler):
    def __init__(self, base_value, final_value, epochs, niter_per_ep, warmup_epochs=10):
        super(WarmupScheduler, self).__init__()
        self.schedule = self.init_schedule(base_value, final_value, epochs, niter_per_ep, warmup_epochs)

    @classmethod
    def init_from_args(cls, args):
        instance = cls(args.lr_start, args.lr_end, args.epochs, args.iters_per_epoch, args.lr_warmup)
        return instance
    
    def init_schedule(self, base_value, final_value, epochs, niter_per_ep, warmup_epochs=10):
        warmup_schedule = np.linspace(base_value, final_value, warmup_epochs * niter_per_ep)
        schedule = np.ones((epochs - warmup_epochs) * niter_per_ep) * final_value

        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == epochs * niter_per_ep
        return schedule

    def get_schedule(self):
        return self.schedule