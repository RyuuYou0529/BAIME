import torch.nn as nn
from abc import ABC, abstractmethod

from ..core.register import ARCH_REGISTER

class BaseModel(nn.Module, ABC):
    def __init__(self):
        super(BaseModel, self).__init__()
    
    @classmethod
    @abstractmethod
    def init_from_args(cls, args):
        raise NotImplementedError

@ARCH_REGISTER.register('ExampleMLP')
class ExampleMLP(BaseModel):
    def __init__(self, input_dim, output_dim):
        super(ExampleMLP, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    @classmethod
    def init_from_args(cls, args):
        instance = cls(args.input_dim, args.output_dim)
        return instance

    def forward(self, x):
        return self.layer(x)