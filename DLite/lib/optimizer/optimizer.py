import torch
import torch.optim as Opt
from abc import ABC, abstractmethod

from ..core.register import OPTIMIZER_REGISTER

class BaseOptimizer(Opt.Optimizer, ABC):
    def __init__(self):
        super(BaseOptimizer, self).__init__()

    @classmethod
    @abstractmethod
    def init_from_args(cls, args, model):
        raise NotImplementedError

@OPTIMIZER_REGISTER.register('adam')
class Adam_Opt(BaseOptimizer, Opt.Adam):
    def __init__(self, model:torch.nn.Module, lr):
        Opt.Adam.__init__(self, model.parameters(), lr=lr)

    @classmethod
    def init_from_args(cls, args, model):
        instance = cls(model, args.lr_start)
        return instance

@OPTIMIZER_REGISTER.register('sgd')
class SGD_Opt(BaseOptimizer, Opt.SGD):
    def __init__(self, model:torch.nn.Module, lr):
        Opt.SGD.__init__(self, model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    @classmethod
    def init_from_args(cls, args, model):
        instance = cls(model, args.lr_start)
        return instance

@OPTIMIZER_REGISTER.register('adagrad')
class Adagrad_Opt(BaseOptimizer, Opt.Adagrad):
    def __init__(self, model:torch.nn.Module, lr):
        Opt.Adagrad.__init__(self, model.parameters(), lr=lr)

    @classmethod
    def init_from_args(cls, args, model):
        instance = cls(model, args.lr_start)
        return instance
