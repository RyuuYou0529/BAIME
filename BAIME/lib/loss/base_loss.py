import torch.nn as nn
from abc import ABC, abstractmethod

from ..core.register import LOSS_REGISTER

class BaseLoss(nn.Module, ABC):
    def __init__(self):
        super(BaseLoss, self).__init__()
    
    @classmethod
    @abstractmethod
    def init_from_args(cls, args):
        raise NotImplementedError

@LOSS_REGISTER.register('ExampleLoss')
class ExampleLoss(BaseLoss):
    def __init__(self, l1_weight=0.5, l2_weight=0.5):
        super(ExampleLoss, self).__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.l1_loss_fn = nn.L1Loss(reduction='mean')
        self.l2_loss_fn = nn.MSELoss(reduction='mean')
        self.loss_logger = dict()
        self.image_logger = dict()

    @classmethod
    def init_from_args(cls, args):
        instance = cls()
        return instance

    def forward(self, preds, labels):
        l1_loss = self.l1_loss_fn(preds, labels)
        l2_loss = self.l2_loss_fn(preds, labels)
        loss = self.l1_weight * l1_loss + self.l2_weight * l2_loss

        self.loss_logger['l1_loss'] = l1_loss.item()
        self.loss_logger['l2_loss'] = l2_loss.item()
        self.loss_logger['total_loss'] = loss.item()

        return loss, self.loss_logger, self.image_logger